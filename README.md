# EMG Hand Gesture Classification

The methodology implemented achieved up to **92% accuracy** in classifying six different hand gestures using a 4-layer MLP. This result is consistent with and slightly surpasses many reported accuracies in the literature, which typically hover around **90%** for EMG-based hand gesture recognition.

One persistent challenge is accurately identifying the **rest state**, which suffers from low signal-to-noise ratio (SNR) due to minimal muscle activation. This limitation affects both this study and prior work, making rest state detection a key area for improvement.

Several recent studies support these observations:

- The study *Research on Gesture Recognition of Surface EMG Based on Machine Learning* reported CNN accuracy of 99.47% and MLP accuracy of 98.42% on nine gestures, showing that deep learning models can achieve very high performance, though challenges remain for rest state classification.
- Other studies typically report accuracy in the 90-95% range on similar multi-class EMG classification tasks, aligning with the 92% accuracy obtained here with a relatively simple MLP model.
- This suggests that with careful preprocessing and model tuning, MLPs can be competitive with more complex CNN architectures.

---

# Methodology

## Data Acquisition and Preprocessing

EMG signals were recorded using the **MYO Thalmic bracelet**, which contains eight equally spaced sensors around the forearm, transmitting data via Bluetooth to a PC.

Preprocessing of raw EMG signals was performed using a MATLAB script. Key preprocessing steps include:

- **Filtering:** A **10th order Butterworth bandpass filter** was implemented with cutoff frequencies between **20 Hz and 450 Hz** to remove motion artifacts, baseline drift, and high-frequency noise outside the typical EMG frequency range.  
- **Segmentation:** Signals were segmented according to gesture labels with pauses between gesture executions to minimize overlap.

These preprocessing steps help enhance the signal quality and improve classifier performance.

---

# EMG Pattern Database

The dataset consists of raw EMG data from **36 subjects** performing a series of static hand gestures. Each subject performed two series, each containing six (sometimes seven) basic gestures. Each gesture was held for 3 seconds, followed by a 3-second rest period.

### Description of raw data files

Each data file contains 10 columns:

1. **Time:** Timestamp in milliseconds  
2-9. **Channels 1-8:** EMG signals from the eight sensors of the MYO Thalmic bracelet  
10. **Class label:** Gesture identifiers  
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

The current model achieves up to **92% accuracy** across six gesture classes:

- Resting  
- Fist clenched  
- Wrist flexion  
- Wrist extension  
- Ulnar deviation  
- Radial deviation  

The confusion matrix below highlights that most misclassifications occur in the resting state, confirming that low SNR in relaxed muscles poses a classification challenge:

![Confusion Matrix](https://github.com/user-attachments/assets/2bfee2e2-6143-49d0-b3e6-bd96f16cf545)

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
