import os
import glob
import torch
import csv
import warnings
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from transformers import AutoTokenizer
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from clip3 import CLIP_MAE  # Ensure this is the updated binary classifier model
from contextlib import redirect_stdout

warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
check_pt = '/scratch/akasturi/src/0_ECMO_files/Phase_2/VLM/CLIP_MAE_V8/P1/S1/clip_mae_v8_s1_all_e7_best_auc_model_acc76_auc85.pth'
n = '1'
version = f"clip_mae_v7_s{n}_p1_all"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the testing dataset
class CPCDataset(Dataset):
    def __init__(self, data_path= f'data/s1', text_path='data/prompt1_day_all',
                 max_text_len=77, phase='test', num_frames_to_sample=16):
        self.data_path = os.path.join(data_path, phase)
        self.num_frames_to_sample = num_frames_to_sample
        self.files = glob.glob(os.path.join(self.data_path, '*/*.mp4'))
        self.text_path = text_path
        # Use BioClinicalBERT tokenizer (or the tokenizer you used during training)
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.max_text_len = max_text_len
        self.video_transform = Compose([
            Resize((224, 224), antialias=True),
            CenterCrop(224),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # For binary classification, labels are 0 or 1 (converted to float)
        self.class_labels = ['0', '1']
        self.label2id = {label: idx for idx, label in enumerate(self.class_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        video_path = self.files[idx]
        ris = video_path.split('/')[-1].split('_')[1]
        video = self.process_video(video_path)
        text_input_ids, attention_mask = self.process_text(ris)
        # Assume label is in the parent folder name
        label = float(int(video_path.split('/')[-2]))
        return {
            "video": video,
            "text_input_ids": text_input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.float32)
        }

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

# Initialize test dataset and loader
test_dataset = CPCDataset(phase='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# Initialize model (with num_classes=1 for binary classification)
model = CLIP_MAE( num_classes=1).to(DEVICE)

# Load the trained checkpoint
checkpoint = torch.load(check_pt, map_location=DEVICE)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

all_labels, all_preds, all_probs = [], [], []

# Open a text file to capture all print outputs
output_file = f"{version}_output.txt"
with open(output_file, "w") as f, redirect_stdout(f):
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Testing', unit='batch')
        for batch in test_bar:
            videos = batch['video'].to(DEVICE)
            text_input_ids = batch['text_input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(videos=videos, input_ids=text_input_ids, attention_mask=attention_mask)
            # For binary classification, apply sigmoid to get probabilities
            logits = outputs['logits']  # shape: (batch_size, 1)
            probs = torch.sigmoid(logits.squeeze(1))
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    # Compute Youden's index for each threshold (Youden = sensitivity + specificity - 1, equivalent to tpr - fpr)
    youden_index = tpr - fpr

    # Select the threshold that maximizes Youden's index
    best_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_idx]
    print(f"Best threshold (maximizing Youden's index): {best_threshold:.4f}")

    # Recompute predictions using the best threshold
    all_probs_np = np.array(all_probs)
    preds_opt = (all_probs_np >= best_threshold).astype(int)

    # Recalculate metrics with the optimized threshold
    accuracy_opt = accuracy_score(all_labels, preds_opt)
    f1_opt = f1_score(all_labels, preds_opt)
    auc_opt = roc_auc_score(all_labels, all_probs)  # AUC remains threshold-independent
    cm_opt = confusion_matrix(all_labels, preds_opt)
    if cm_opt.size == 4:
        tn, fp, fn, tp = cm_opt.ravel()
    else:
        tn = fp = fn = tp = 0
    sensitivity_opt = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity_opt = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\nOptimized Metrics using the best threshold:")
    print(f"Accuracy:        {accuracy_opt:.4f}")
    print(f"F1 Score:        {f1_opt:.4f}")
    print(f"AUC:             {auc_opt:.4f}")
    print(f"Sensitivity:     {sensitivity_opt:.4f}")
    print(f"Specificity:     {specificity_opt:.4f}")
    print("Confusion Matrix:")
    print(cm_opt)

    # (Optional) Plot the ROC curve with the best threshold marked
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_opt:.4f})')
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f'Best Threshold = {best_threshold:.4f}')
    plt.plot([0, 1], [0, 1], 'r--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Optimized Threshold')
    plt.legend(loc='lower right')
    plt.savefig(f"{version}_roc_curve_optimized.png")
    plt.show()
