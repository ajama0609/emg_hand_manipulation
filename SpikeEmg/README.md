# SpikeEMG

## Introduction  
This work reproduces and extends results from EmgHandNet using the NinaPro DB1 dataset. Leveraging a hybrid **Artificial Neural Network (ANN)** and **Spiking Neural Network (SNN)** architecture with just **42,893 parameters**, the proposed model achieves a test accuracy of **94.19%** (macro F1-score: **0.9441**) with an **average inference time of 1.3 µs per sample** on an Nvidia RTX 4070 (8GB VRAM).

This design offers a low-latency, energy-efficient solution suitable for real-time prosthetic control and embedded biomedical systems.

---

## Methodology  

### Preprocessing  
Preprocessing was conducted in **MATLAB** and **Python**:

- **Filtering:** Applied a 10th-order high-pass filter at 15 Hz. EMG signals primarily reside between 20–500 Hz.  
- **Time-domain features:**  
  - Root Mean Square (RMS)  
  - Mean Absolute Value (MAV)  
  - Variance (VAR)  
- **Frequency-domain features:**  
  - FFT power  
  - Peak frequency  

Python pipeline:  
- Applied **SMOTE** for class balancing  
- Dataset split: 80% training / 10% validation / 10% testing  
- Reshaped feature vectors to include a **temporal dimension** compatible with the **Leaky Integrate-and-Fire (LIF)** spiking layer.

---

## Model Architecture  

A hybrid ANN-SNN architecture was used:

| Layer                  | Output Shape | Parameters |
|------------------------|--------------|------------|
| Linear-1               | [-1, 128]    | 6,528      |
| Linear-2               | [-1, 256]    | 33,024     |
| PiecewiseLeakyReLU-3   | [-1, 256]    | 0          |
| LIFNode-4              | [-1, 256]    | 0          |
| Linear-5               | [-1, 13]     | 3,341      |

- **Total parameters:** 42,893  
- **Memory footprint:** ~0.17 MB  
- **Encoding:** None used. Temporal richness of EMG features was sufficient; spike encoding **reduced** performance.

---

## How to Use This Resource  

1. **Preprocessing:**  
   - Run `Preprocessing.m` in the `s1/` folder to generate `.csv` feature files.  

2. **Training & Evaluation:**  
   - Use the Python scripts in `SpikeEMG/` for training and evaluation.  
   - Install dependencies with:
     ```bash
     pip install -r requirements.txt
     ```

3. **Output:**  
   - Logs, metrics, and model checkpoints are saved in the `logs/` directory.  

---

## Evaluation  

| Metric                 | Value       |
|------------------------|-------------|
| **Test Accuracy**      | **94.19%**  |
| **Macro F1-score**     | **0.9441**  |
| **Weighted F1-score**  | **0.9419**  |
| **Inference Time**     | **1.3 µs/sample** |
| **Model Size**         | **0.17 MB** |
| **Total Parameters**   | **42,893**  |

The model significantly outperforms traditional classifiers (e.g., SVM: 70–85%) and deep EMG networks like EmgHandNet (80–90%), while using **under 50K parameters** and achieving **real-time inference**.

---

## Notes & Future Work  
- Future directions include **multimodal fusion** with IMU or EEG data.  
- The lightweight architecture is promising for **on-device prosthetics**, **wearables**, or **neuromorphic edge computing**.  
- Full training logs and confusion matrices are included in the repository.

---

## Example Training Command

```bash
python train.py --config configs/spikeemg.yaml
