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
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
import torch.nn.functional as F
from clip3 import CLIP_MAE  # Ensure this is the model defined above

warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
version = "clip_mae_v8_s1_all"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 20
NUM_EPOCHS = 11
GRAD_ACCUM_STEPS = 1
best_test_auc = 0.0
# Dataset Definition
class CPCDataset(Dataset):
    def __init__(self, data_path='data/v11', text_path='data/prompt1_day_all',
                 max_text_len=77, phase='train', num_frames_to_sample=16):
        self.data_path = os.path.join(data_path, phase)
        self.num_frames_to_sample = num_frames_to_sample
        self.files = glob.glob(os.path.join(self.data_path, '*/*.mp4'))
        self.text_path = text_path
        # Use BioClinicalBERT tokenizer
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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            video_path = self.files[idx]
            ris = video_path.split('/')[-1].split('_')[1]
            
            # Process video and check for NaNs
            video = self.process_video(video_path)
            if torch.isnan(video).any():
                raise ValueError(f"NaN detected in video: {video_path}")
            
            # Process text
            text_input_ids, attention_mask = self.process_text(ris)
            
            # Get label from directory name and convert to float for BCE loss
            label = float(int(video_path.split('/')[-2]))
            return {
                "video": video,
                "text_input_ids": text_input_ids,
                "attention_mask": attention_mask,
                "labels": torch.tensor(label, dtype=torch.float32)
            }
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            # Retry with next index
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
        # Apply transform to each frame and stack them
        return torch.stack([self.video_transform(frame.permute(2, 0, 1) / 255.0) for frame in video])

# Create datasets and loaders
train_dataset = CPCDataset(phase='train')
val_dataset = CPCDataset(phase='val')
test_dataset = CPCDataset(phase='test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Initialize model, optimizer, scheduler, and scaler
model = CLIP_MAE(num_classes=1, freeze_backbones=True).to(DEVICE)
scaler = GradScaler()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=2e-4,
    total_steps=NUM_EPOCHS * (len(train_loader) // GRAD_ACCUM_STEPS),
    pct_start=0.1
)

# CSV logging for training metrics
csv_file = f"{version}_training_logs.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 
                     'val_acc', 'val_f1', 'val_auc', 'best_acc', 'lr', 'timestamp'])

# Independent best metrics tracking
best_acc = 0.0
best_auc = 0.0
best_auc_model_path = None  # To store the best AUC model file path

def evaluate_on_test(model, test_loader, device):
    """
    Evaluate the model on the test set and compute the AUC.
    """
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Evaluation"):
            videos = batch["video"].to(device)
            text_ids = batch["text_input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(videos, text_ids, mask, labels)
            all_logits.append(outputs["logits"])
            all_labels.append(labels)
    
    if all_logits:
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        probs = torch.sigmoid(logits.squeeze(1))
        test_auc = roc_auc_score(labels.cpu(), probs.cpu().detach().numpy())
    else:
        test_auc = 0.0
    return test_auc

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    for i, batch in pbar:
        try:
            videos = batch["video"].to(DEVICE)
            text_ids = batch["text_input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            with autocast():
                outputs = model(videos, text_ids, mask, labels)
                loss = outputs["loss"] / GRAD_ACCUM_STEPS
                
                if torch.isnan(loss):
                    print("NaN loss detected, skipping batch")
                    continue
                    
            scaler.scale(loss).backward()
            
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            train_loss += loss.item() * GRAD_ACCUM_STEPS
            
            preds = (torch.sigmoid(outputs["logits"].squeeze(1)) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f"{loss.item()*GRAD_ACCUM_STEPS:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                'best_auc': f"{best_auc:.4f}"
            })
        except Exception as e:
            print(f"\nError in batch {i}: {str(e)}")
            continue
            
    avg_train_loss = train_loss / len(train_loader)
    train_acc = correct / total if total > 0 else 0.0
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    all_logits = []
    all_labels = []
    
    pbar_val = tqdm(val_loader, total=len(val_loader),
                    desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
    for batch in pbar_val:
        videos = batch["video"].to(DEVICE)
        text_ids = batch["text_input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        outputs = model(videos, text_ids, mask, labels)
        if torch.isnan(outputs["loss"]):
            print("NaN validation loss detected, skipping batch")
            continue
                
        val_loss += outputs["loss"].item()
        all_logits.append(outputs["logits"])
        all_labels.append(labels)
        pbar_val.set_postfix({
            'val_loss': f"{outputs['loss'].item():.4f}",
            'best_acc': f"{best_acc:.4f}"
        })
    
    if all_logits:
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        probs = torch.sigmoid(logits.squeeze(1))
        preds = (probs > 0.5).float()
    
        val_acc = (preds == labels).float().mean().item()
        val_f1 = f1_score(labels.cpu(), preds.cpu())
        val_auc = roc_auc_score(labels.cpu(), probs.cpu().detach().numpy())
    else:
        val_acc = val_f1 = val_auc = 0.0

    avg_val_loss = val_loss / len(val_loader)
    
    # Save best accuracy model
    if val_acc > best_acc:
        best_acc = val_acc
        acc_str = f"{int(val_acc * 100):02d}"
        auc_str = f"{int(val_auc * 100):02d}"
        torch.save(model.state_dict(), f"{version}_e{epoch+1}_best_acc_model_acc{acc_str}_auc{auc_str}.pth")
    
    # Save best AUC model independently and evaluate on test set when improved
    if val_auc > best_auc:
        best_auc = val_auc
        acc_str = f"{int(val_acc * 100):02d}"
        auc_str = f"{int(val_auc * 100):02d}"
        best_auc_model_path = f"{version}_e{epoch+1}_best_auc_model_acc{acc_str}_auc{auc_str}.pth"
        torch.save(model.state_dict(), best_auc_model_path)
        
        # Evaluate test set when best AUC is updated
        test_auc = evaluate_on_test(model, test_loader, DEVICE)
        best_test_auc = test_auc
        print(f"\nEpoch {epoch+1}: Updated Best AUC Model - Test AUC: {test_auc:.4f}")
    
    # Log metrics to CSV
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch+1,
            avg_train_loss,
            avg_val_loss,
            train_acc,
            val_acc,
            val_f1,
            val_auc,
            best_acc,
            scheduler.get_last_lr()[0],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])
    
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    print(f"Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
    print(f"Best Test Auc so far: {best_test_auc:.4f}")

print("Training Complete!")
print(f"Best Metrics - Acc: {best_acc:.4f}, AUC: {best_auc:.4f}")
