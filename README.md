# Human Emotion Recognition from Facial Images using Deep Learning  
*End-to-End Benchmarking of CNNs, EfficientNet, Vision Transformers, and Hugging Face ViT*

---

## Project Overview

Accurately detecting human emotions from facial images is a challenging and high-impact problem in AI, powering applications in mental health, affective computing, human-computer interaction, and more. In this project, I systematically **benchmark classic and state-of-the-art deep learning models** to classify facial emotions (`happy`, `sad`, `angry`) from images. My approach blends the best of modern computer vision: from robust convolutional architectures (ResNet, EfficientNet, VGG) to powerful transformer-based models (custom ViT and Hugging Face’s pre-trained ViT).

This repository demonstrates a *research-grade, production-ready workflow*: starting with data wrangling and class balancing, moving through exploratory analysis, multiple modeling strategies, interpretability (feature maps, GradCAM), and ending with an ensemble and a SOTA ViT solution.

---

## Dataset
https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes/code

The project leverages a curated facial emotion dataset (Kaggle), containing labeled RGB images of faces expressing core emotions.  
- **Classes:** `happy`, `sad`, `angry`
- **Sample Size:** ~6,800 images, with class imbalance
- **Input Format:** Color face images, variable backgrounds and lighting

### Features
- **Image Data:** Raw RGB facial images
- **Labels:** One-hot encoded emotion class

---

## Process Overview

The project is a journey of **iterative experimentation and improvement**:

- **Data Cleaning & Augmentation:** Outlier removal, face detection, brightness/rotation/contrast augmentation.
- **Exploratory Data Analysis (EDA):** Class distribution, sample visualization, and batch-wise mean face.
- **Class Balancing:** Computed and applied per-class weights to handle underrepresented emotions.
- **Modeling:** Stepwise benchmarking of progressively advanced deep learning architectures (ResNet-34 → EfficientNetB4 → VGG-16 for interpretability → Custom Vision Transformer → Hugging Face ViT).
- **Interpretability:** Feature map visualization, GradCAM heatmaps, patch analysis for ViTs.
- **Evaluation:** Accuracy, loss, confusion matrices, class-wise recall, and cross-architecture comparison.

---

## Models Explored

### ResNet-34  
- **What:** 34-layer convolutional neural network with skip connections (deep residual learning).
- **Why:** Robust baseline, proven for image classification.  
- **Result:** *Validation Accuracy: ~52%*. Provided a solid starting point, but struggled with subtle expressions and class imbalance.

### EfficientNetB4  
- **What:** Compound-scaled, parameter-efficient CNN with squeeze-and-excitation and advanced scaling.
- **Why:** Top performer on ImageNet, great for transfer learning with limited data.
- **Result:** *Validation Accuracy: ~76%*. Large jump in performance, especially after unfreezing and fine-tuning all layers with class weights.

### VGG-16  
- **What:** Classic deep CNN with uniform 3x3 convolutions; very interpretable.
- **Why:** Used for feature map and activation visualization to demystify CNN decisions.
- **Result:** Provided rich feature insights, guiding downstream improvements.

### Custom Vision Transformer (ViT)  
- **What:** Patch-based self-attention model built from scratch; transformer encoder blocks process image patches as “tokens.”
- **Why:** To explore the power and limitations of transformers in vision without pretraining.
- **Result:** *Validation Accuracy: ~44%*. Showed the promise and data-hunger of ViTs—highlighted the need for large-scale pretraining.

### Hugging Face ViT (`google/vit-base-patch16-224-in21k`)  
- **What:** State-of-the-art, pre-trained Vision Transformer, fine-tuned for my emotion dataset.
- **Why:** Leverages massive pretraining for world-class accuracy, quick convergence, and strong generalization.
- **Result:** **Validation/Test Accuracy: 96.9%**. Minimal confusion, robust across all classes, and production-ready.

---

## Exploratory Data Analysis (EDA)

- **Class Balance:** Dataset had significant imbalance (happy: 3k+, sad: 2.2k, angry: 1.5k).
- **Batch Visualizations:** Displayed random batches, checked for annotation noise/outliers.
- **Feature Maps:** Visualized how each model’s early, mid, and deep layers responded to emotional cues—mouth, eyes, eyebrows.

---

## Feature Engineering & Preprocessing

- **Data Augmentation:** Controlled rotation, contrast, horizontal flip, brightness—improved robustness.
- **Class Weights:** Applied per-class weights in loss function, calculated from dataset frequencies. Boosted recall for minority classes (`angry`, `sad`).
- **Normalization:** Images scaled to [0, 1].
- **Patch Extraction:** For ViT models, each image was split into 16x16 patches and encoded for transformer processing.

---

## Modeling and Benchmarking

- **Training/Validation Split:** Stratified 80/20, preserving class proportions.
- **Early Stopping & Checkpoints:** Used to prevent overfitting and select best models.
- **Experiment Tracking:** Integrated [Weights & Biases](https://wandb.ai/) for full reproducibility.

### Performance Table

| Model                   | Type             | Validation Accuracy | Comments                       |
|-------------------------|------------------|---------------------|--------------------------------|
| ResNet-34               | Deep CNN         | ~52%                | Struggled on minority classes  |
| EfficientNetB4          | TL, fine-tuned   | ~76%                | Major boost; robust features   |
| Custom Vision Transformer| Transformer     | ~44%                | Needed more data/pretraining   |
| Ensemble (ResNet+EffNet)| Averaged         | ~84%                | Boosted stability & recall     |
| **Hugging Face ViT**    | Pre-trained ViT  | **96.9%**           | SOTA, best all-round performer |

---

## Evaluation & Interpretability

- **Confusion Matrix:** Hugging Face ViT achieved nearly perfect class separation.
- **GradCAM & Patch Attention:** Visualized where models focused for each emotion (e.g., smiles for happy, eyebrows for angry).
- **Per-Class Recall:** Noted large improvements for minority classes after applying class weights.

---

## Key Takeaways

- **SOTA Models Matter:** Pretrained transformers, when fine-tuned, dramatically outperform both CNN baselines and scratch-built models—even on modest data.
- **Class Weights are Critical:** Imbalanced data *requires* weighted loss for reliable results.
- **Interpretability Builds Trust:** Feature maps and GradCAM make “black box” models transparent.
- **Experimentation Culture:** Iterative testing, careful evaluation, and critical error analysis drive results.

---

## How to Reproduce

1. **Clone the repo** and install requirements:  
   `pip install -r requirements.txt`
2. Place images in `/data/`, organized by emotion class.
3. Train or evaluate any model via Jupyter notebook or script.
4. For best results, run the Hugging Face ViT notebook with class weighting enabled.

---

## Why This Project?

This project reflects my approach as a **data scientist**:  
- *Curious and methodical*: Benchmarking, not just picking a “favorite” model  
- *Research-driven*: Adapting the very latest SOTA architectures  
- *Focused on real-world impact*: Solutions ready for production, robust to data challenges, and transparent for stakeholders

