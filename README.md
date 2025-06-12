# CLAIR-MAE: Contrastive Language and Image Reasoning with Masked Autoencoders

Multimodal deep learning approach for **early neurological prognostication** of post-cardiac arrest patients

**Authors:**  
Akhil Kasturi¹*, Axel Wismüller¹,³  
¹ Dept. of ECE, Univ. of Rochester • 


## 🎯 Overview

CLAIR-MAE uses cross-attention between CT volumes and clinical features to predict 24-h neurological outcome (CPC) with  
**AUC-ROC = 0.94**—outperforming standalone imaging or clinician assessments.

## ⚙️ Getting Started

### Prerequisites

- Python 3.8+  
- PyTorch ≥1.10  
- `transformers`, `pydicom`, `scikit-learn`

### Installation

```bash
git clone https://github.com/yourusername/clair-mae-prognosticator.git
cd clair-mae-prognosticator
pip install -r requirements.txt
