# RetinexFormer-LLIE  
### Joint Low-Light Enhancement and Residual Denoising (Wild Scene Dataset)

This repository implements a **RetinexFormer-style hybrid model** for joint low-light image enhancement and residual denoising.  

It is designed for large-scale datasets (10,000+ images) captured in real-world wild scenes, where perceptual quality is prioritized over pure noise suppression.

---

## 📌 Overview

Low-light images often suffer from:
- Illumination imbalance
- Reduced contrast
- Color shifts
- Residual sensor noise

This model follows a **Retinex-based decomposition + Transformer enhancement + lightweight noise refinement** pipeline:

```
Input RGB
    ↓
Retinex Decomposition
    ├── Illumination Map
    └── Reflectance Map
    ↓
Transformer-based Illumination Enhancement
    ↓
Reflectance Noise Refinement
    ↓
Reconstruction (Enhanced RGB)
```

---

## 🚀 Features

- Retinex decomposition (Illumination + Reflectance)
- Transformer-based global illumination modeling
- Lightweight residual noise refinement
- Mixed precision (AMP) training
- Resume training support
- Automatic best-model saving
- Training on 10,000-image subset
- Folder-based batch testing
- PSNR & SSIM evaluation

---

## 📂 Project Structure

```
retinexformer_llie/
│
├── train.py
├── test.py
├── evaluate.py
├── requirements.txt
│
├── models/
│   ├── retinexformer.py
│   ├── transformer_block.py
│   ├── decomposition.py
│   ├── refinement.py
│
├── losses/
│   ├── losses.py
│
├── datasets/
│   ├── llie_dataset.py
│
├── utils/
│   ├── metrics.py
│
└── checkpoints/
```

---

## 🛠 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your_username/retinexformer_llie.git
cd retinexformer_llie
```

### 2️⃣ Create environment

```bash
conda create -n llie python=3.10 -y
conda activate llie
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Structure

Training dataset must be structured as:

```
data/
├── low/
├── high/
├── val_low/
└── val_high/
```

- `low/` → low-light input images
- `high/` → reference images
- `val_low/` → validation inputs
- `val_high/` → validation references

Images must be paired and sorted consistently.

---

## 🏋️ Training

### Train on 10,000 images

```bash
python train.py
```

If dataset contains more than 10,000 images, a random subset of 10,000 will be used automatically.

### Resume Training

Edit inside `train.py`:

```python
RESUME_PATH = "checkpoints/last.pth"
```

Then run:

```bash
python train.py
```

---

## 💾 Checkpoint Saving

- `checkpoints/last.pth` → latest checkpoint (every epoch)
- `checkpoints/best_model.pth` → best model (based on validation PSNR)

Best model is updated automatically when validation PSNR improves.

---

## 🧪 Testing on a Folder

Place test images inside:

```
test_images/
```

Then run:

```bash
python test.py
```

Results will be saved in:

```
results/
```

---

## 📈 Evaluation Metrics

The model is evaluated using:

### 🔹 PSNR (Peak Signal-to-Noise Ratio)

Measures reconstruction quality relative to ground truth.

Higher is better.

### 🔹 SSIM (Structural Similarity Index)

Measures structural similarity and perceptual consistency.

Range: 0 – 1  
Higher is better.

---

## 🧠 Architecture Details

### Retinex Decomposition

Separates input image into:

- Illumination (brightness information)
- Reflectance (texture & structure)

### Illumination Enhancement

- Multi-head self-attention transformer blocks
- Global context modeling
- Adaptive brightness correction

### Reflectance Refinement

- Lightweight residual CNN
- Removes mild residual noise
- Preserves texture

### Reconstruction

```
Enhanced Output = Enhanced Illumination × Refined Reflectance
```

---

## 🎯 Design Philosophy

This model is designed for:

- Wild-scene low-light datasets
- Mild residual noise
- Perceptual quality optimization
- Competition-level benchmarks

Heavy denoising is intentionally avoided to preserve texture realism.

---

## ⚡ Future Improvements

- Multi-scale transformer
- EMA model tracking
- Distributed training (DDP)
- Cosine learning rate scheduler
- LPIPS perceptual optimization
- Window-based attention (Swin-style)
- Patch-based 512×512 training

---

## 📜 License

This project is provided for research and academic purposes.

---

## 👤 Author

Your Name  
GitHub: https://github.com/your_username  

---

## ⭐ If This Helps You

If this repository helps your research or competition submission, please consider giving it a star ⭐
