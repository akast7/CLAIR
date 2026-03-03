import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEModel, AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

class CLIP_MAE(nn.Module):
    def __init__(
        self,
        vision_encoder="MCG-NJU/videomae-large-finetuned-kinetics",
        text_encoder="emilyalsentzer/Bio_ClinicalBERT",  # Use BioClinicalBERT
        freeze_backbones=True,
        num_classes=1,  # Single output for binary classification
        use_lora=True
    ):
        super().__init__()

        # Vision Encoder
        self.vision_encoder = VideoMAEModel.from_pretrained(vision_encoder)
        if freeze_backbones:
            for param in self.vision_encoder.parameters():
                param.requires_grad_(False)
        
        # Text Encoder using BioClinicalBERT with LoRA adaptation if desired
        self.text_encoder = AutoModel.from_pretrained(text_encoder)
        if freeze_backbones:
            for param in self.text_encoder.parameters():
                param.requires_grad_(False)
        
        # Projections with Dropout for regularization
        self.vision_proj = nn.Sequential(
            nn.Linear(self.vision_encoder.config.hidden_size, 768),
            nn.LayerNorm(768),
            nn.Dropout(0.1)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, 768),
            nn.LayerNorm(768),
            nn.Dropout(0.1)
        )

        # Cross-Attention Layers with residual dropout
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(768, num_heads=8, batch_first=True)
            for _ in range(3)
        ])
        self.norm = nn.LayerNorm(768)
        self.residual_dropout = nn.Dropout(0.1)

        # Temporal Pooling
        self.temporal_pool = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8),
            num_layers=2
        )

        # Classifier outputs a single logit per sample
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Loss parameters (for stability)
        self.logit_scale = nn.Parameter(torch.tensor(0.07))
        self.loss_alpha = nn.Parameter(torch.tensor(0.5))

        # Optional: Add LoRA adaptation to text encoder if desired
        if use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules=["query", "value"],
                lora_dropout=0.2,
                modules_to_save=["classifier"]
            )
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)

    def forward(self, videos, input_ids, attention_mask, labels=None):
        # Extract video features
        vision_outputs = self.vision_encoder(videos)
        vision_features = self.vision_proj(vision_outputs.last_hidden_state)
        vision_features = self.temporal_pool(vision_features).mean(1)

        # Extract text features using BioClinicalBERT (with attention mask)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_proj(text_outputs.last_hidden_state[:, 0, :])

        # Cross-Attention Fusion with residual dropout and scaling
        for attn_layer in self.cross_attn:
            attn_output, _ = attn_layer(
                query=text_features.unsqueeze(1),
                key=vision_features.unsqueeze(1),
                value=vision_features.unsqueeze(1)
            )
            attn_output = attn_output.squeeze(1)
            attn_output = self.residual_dropout(attn_output)
            text_features = self.norm(text_features + 0.5 * attn_output)

        # Classifier: output shape (batch_size, 1)
        logits = self.classifier(text_features)

        loss = None
        if labels is not None:
            # Normalize features for contrastive loss stability
            vision_norm = F.normalize(vision_features, dim=-1, eps=1e-6)
            text_norm = F.normalize(text_features, dim=-1, eps=1e-6)
            batch_size = labels.shape[0]
            targets = torch.arange(batch_size, device=vision_features.device)
            logit_scale_clamped = torch.clamp(self.logit_scale.exp(), max=50.0)
            logits_per_video = (text_norm @ vision_norm.T) * logit_scale_clamped

            contrastive_loss = (
                F.cross_entropy(logits_per_video, targets) +
                F.cross_entropy(logits_per_video.T, targets)
            ) / 2

            # Binary classification loss (BCE with logits)
            cls_loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), labels.float())
            adaptive_weight = torch.clamp(self.loss_alpha.sigmoid(), min=0.1, max=0.9)
            loss = adaptive_weight * contrastive_loss + (1 - adaptive_weight) * cls_loss

        return {"loss": loss, "logits": logits}
