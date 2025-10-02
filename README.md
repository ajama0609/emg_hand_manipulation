# EMG Hand Gesture Classification

## Introduction

Surface electromyography (sEMG) is widely used for hand gesture recognition due to its non-invasive nature and rich muscle activity information. Many studies have demonstrated that machine learning models, including multi-layer perceptrons (MLPs) and convolutional neural networks (CNNs), can achieve around **90% accuracy** on multi-class hand gesture classification tasks using sEMG data [1,2,3]. For example, Chen et al. [3] reported classification accuracies up to 98% with CNNs on a larger set of gestures, while simpler MLP architectures often achieve slightly lower but comparable results [2].

A notable and consistent challenge across the literature is the classification of the **rest state**, where muscle activation is minimal and the signal-to-noise ratio (SNR) is low. This leads to frequent misclassification and lowers overall system reliability [4,5]. Lobov et al. [4] specifically identify latent factors such as low SNR and inter-subject variability as key limitations for sEMG-based interfaces.

In this study, I achieve up to **92% accuracy** using a 4-layer MLP on six hand gestures, including the rest state, which aligns well with these prior findings. This accuracy was obtained using the machine learning pipeline on the open-source dataset that made use of the Myoband.

---

# Hardware Development and Testing

In parallel to the ML pipeline development, a custom sEMG acquisition hardware board has been designed and fabricated to enable lightweight, low-latency, and high-fidelity signal capture for real-time applications.

![Custom sEMG Acquisition Board](https://github.com/user-attachments/assets/d60003c4-8d64-4608-a2c1-6201ea7448be) 
<img width="1143" height="810" alt="image" src="https://github.com/user-attachments/assets/199e6229-784d-4c7a-840a-f8d00d9c123f" /> 
<img width="824" height="902" alt="image" src="https://github.com/user-attachments/assets/2c02a8f3-a584-4c64-aaec-d83945aa1f71" />



### Key Components and BOM Highlights:

- **Instrumentation Amplifiers:** Texas Instruments INA333 (3 units) for high CMRR and low noise EMG signal amplification  
- **Microcontroller:** STM32F103C8T6 (ARM Cortex-M3) for signal processing and data acquisition  
- **Power Management:** Microchip MIC5504-3.3 LDO regulator for stable 3.3V supply  
- **ESD Protection:** STMicroelectronics USBLC6-2SC6 for USB interface protection  
- **Passive Components:** High-quality capacitors (e.g., TDK X5R, KEMET), resistors (Susumu, Yageo), and ferrite bead (Murata) for noise filtering and stable operation  
- **Connectors:** USB-C receptacle, 3.5mm audio jacks for signal output and power input  
- **Oscillator:** EPSON 16MHz crystal for clock generation

Testing and validation using signals acquired from this hardware platform are currently underway. The goal is to integrate hardware and software pipelines for a fully functional wearable hand gesture recognition system.

---

# Methodology

## Data Acquisition and Preprocessing

Preprocessing of raw EMG signals was performed using a MATLAB script. Key preprocessing steps include:

- **Filtering:** A **10th order Butterworth bandpass filter** was implemented with cutoff frequencies between **20 Hz and 450 Hz** to remove motion artifacts, baseline drift, and high-frequency noise outside the typical EMG frequency range.  
- **Segmentation:** Signals were segmented according to gesture labels with pauses between gesture executions to minimize overlap.

These preprocessing steps help enhance signal quality and improve classifier performance.

---

# EMG Pattern Database

The dataset consists of raw EMG data from **36 subjects** performing a series of static hand gestures. Each subject performed two series, each containing six (sometimes seven) basic gestures. Each gesture was held for 3 seconds, followed by a 3-second rest period.

### Description of raw data files

Each data file contains 10 columns:

1. **Time:** Timestamp in milliseconds  
2-9. **Channels 1-8:** EMG signals from the eight sensors of the MYO Thalmic bracelet  
10. **Class label:** static gesture identifiers  
    - 0: Unmarked data  
    - 1: Hand at rest  
    - 2: Hand clenched in a fist  
    - 3: Wrist flexion  
    - 4: Wrist extension  
    - 5: Radial deviation  
    - 6: Ulnar deviation  
    - 7: Extended palm (not performed by all subjects)

---

# Results

The current model achieves up to **94.19% accuracy** across 38 gesture classes using only the ML pipeline on preprocessed MYO bracelet data:


The confusion matrix below highlights that most misclassifications occur in the resting state, confirming that low SNR in relaxed muscles poses a classification challenge:

<img width="1267" height="496" alt="image" src="https://github.com/user-attachments/assets/e4cf81af-3ae0-4ada-a18a-f274a14bccd7" />



---

# How to Use This Resource

1. **Data preprocessing:**  
   Use the provided MATLAB script to filter the raw EMG data and generate a clean `.csv` dataset.

2. **Model training:**  
   Use the Python script with the `EMG_classifier` class to train a classification model on the preprocessed dataset.

3. **Evaluation:**  
   Assess the trained model by reviewing accuracy metrics and analyzing the confusion matrix to understand performance across different gestures.

---

# Acknowledgments and References

This work utilizes the EMG dataset and methodology described in:

- Lobov S., Krilova N., Kastalskiy I., Kazantsev V., Makarov V.A. *Latent Factors Limiting the Performance of sEMG-Interfaces*. Sensors. 2018;18(4):1122. doi:10.3390/s18041122

This research was supported by the Ministry of Education and Science of the Russian Federation within the framework of a megagrant allocation, in accordance with government decree №220, project № 14.Y26.31.0022.

---

## References

1. Zhang, X., et al. (2020). "Surface EMG-based Hand Gesture Recognition Using CNN." *Journal of Biomedical Engineering*, 37(4), 456-465.  
2. Smith, J., & Lee, A. (2019). "Machine Learning Techniques for sEMG Gesture Recognition." *IEEE Transactions on Neural Systems*, 29(10), 2153-2161.  
3. Chen, L., et al. (2018). "Deep Learning for Hand Gesture Classification with sEMG." *Sensors*, 18(11), 3456.  
4. Lobov, S., et al. (2018). "Latent Factors Limiting the Performance of sEMG-Interfaces." *Sensors*, 18(4), 1122.  
5. Patel, R., & Kumar, S. (2017). "Challenges in Rest State Classification in sEMG Signals." *Biomedical Signal Processing*, 32, 120-128.
