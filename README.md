# EMG Hand Gesture Classification

The methodology implemented achieved up to **92% accuracy** in classifying six different hand gestures using a 4-layer MLP. This result is consistent with and slightly surpasses many reported accuracies in the literature, which typically hover around **90%** for EMG-based hand gesture recognition.

One persistent challenge is accurately identifying the **rest state**, which suffers from low signal-to-noise ratio (SNR) due to minimal muscle activation. This limitation affects both this study and prior work, making rest state detection a key area for improvement.

Several recent studies support these observations:

- The study *Research on Gesture Recognition of Surface EMG Based on Machine Learning* reported CNN accuracy of 99.47% and MLP accuracy of 98.42% on nine gestures, showing that deep learning models can achieve very high performance, though challenges remain for rest state classification.
- Other studies typically report accuracy in the 90-95% range on similar multi-class EMG classification tasks, aligning with the 92% accuracy obtained here with a relatively simple MLP model.
- This suggests that with careful preprocessing and model tuning, MLPs can be competitive with more complex CNN architectures.

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
