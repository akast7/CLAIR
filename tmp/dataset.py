"""
Folder structure - 

data/
  s1/                 # fold 1
    0/                # label = good (example)
      train/*.mp4
      val/*.mp4
      test/*.mp4
    1/                # label = poor
      train/*.mp4
      val/*.mp4
      test/*.mp4
  s2/
    0/...
    1/...
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from transformers import AutoTokenizer


class CPCDataset(Dataset):
    def __init__(self, data_path, text_path, max_text_len, phase, num_frames_to_sample, 
                 tokenizer_model, img_size, img_mean, img_std):
        self.data_path = os.path.join(data_path, phase)
        self.num_frames_to_sample = num_frames_to_sample
        self.files = glob.glob(os.path.join(self.data_path, '*/*.mp4'))
        self.text_path = text_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.max_text_len = max_text_len
        self.video_transform = Compose([
            Resize((img_size, img_size), antialias=True),
            CenterCrop(img_size),
            Normalize(mean=img_mean, std=img_std),
        ])
        self.class_labels = ['0', '1']
        self.label2id = {label: idx for idx, label in enumerate(self.class_labels)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            video_path = self.files[idx]
            ris = video_path.split('/')[-1].split('_')[1]
            
            video = self.process_video(video_path)
            if torch.isnan(video).any():
                raise ValueError(f"NaN detected in video: {video_path}")
            
            text_input_ids, attention_mask = self.process_text(ris)
            
            label = float(int(video_path.split('/')[-2]))
            return {
                "video": video,
                "text_input_ids": text_input_ids,
                "attention_mask": attention_mask,
                "labels": torch.tensor(label, dtype=torch.float32)
            }
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

    def process_text(self, ris):
        text_path = os.path.join(self.text_path, f"{ris}.txt")
        with open(text_path, 'r') as f:
            text = f.read().strip()
        encoded = self.tokenizer(text, padding='max_length', max_length=self.max_text_len,
                                   truncation=True, return_tensors="pt")
        return encoded.input_ids.squeeze(0), encoded.attention_mask.squeeze(0)

    def process_video(self, video_path):
        video_frames, _, _ = read_video(video_path, pts_unit='sec')
        total_frames = video_frames.shape[0]
        start_frame = max(total_frames // 2 - self.num_frames_to_sample // 2, 0)
        end_frame = start_frame + self.num_frames_to_sample
        video = video_frames[start_frame:end_frame]
        if video.shape[0] < self.num_frames_to_sample:
            padding = video[-1:].repeat(self.num_frames_to_sample - video.shape[0], 1, 1, 1)
            video = torch.cat([video, padding], dim=0)
        return torch.stack([self.video_transform(frame.permute(2, 0, 1) / 255.0) for frame in video])
