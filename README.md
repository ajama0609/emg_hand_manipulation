# EMG Hand Gesture Classification

The methodology implemented achieved **89.5% accuracy** in classifying six different hand gestures. This result aligns closely with existing research in the field, which typically reports around **90% classification accuracy** for hand gesture recognition using EMG signals.

One of the key challenges remains the accurate identification of the **rest state**. As illustrated in the image below, distinguishing the resting condition is difficult due to the low signal-to-noise ratio (SNR) when muscles are relaxed and not sending strong signals. This limitation affects both current research and practical applications.

---

# Results

The current model achieves **89.5% accuracy** across six gesture classes:

- Resting  
- Fist clenched  
- Wrist flexion  
- Wrist extension  
- Ulnar deviation  
- Radial deviation  

The most challenging aspect is detecting the **resting state** because the muscle signals are weak, leading to low SNR and lower classification confidence. Improving rest state identification is key to enhancing overall accuracy.

Many studies in the literature achieve around **90% accuracy** in multi-class hand gesture classification **without** employing deep neural networks.

Below is the confusion matrix from the best performing training run:

![Confusion Matrix](https://github.com/user-attachments/assets/2bfee2e2-6143-49d0-b3e6-bd96f16cf545)




---

# How to Use This Resource

1. **Data preprocessing:**  
   Use the provided MATLAB script to filter the raw EMG data and generate a clean `.csv` dataset.

2. **Model training:**  
   Use the Python script with the `EMG_classifier` class to train a classification model on the preprocessed dataset.

3. **Evaluation:**  
   Assess the trained model by reviewing accuracy metrics and analyzing the confusion matrix to understand performance across different gestures.
