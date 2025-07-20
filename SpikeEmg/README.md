# SpikeEMG

## Introduction  
In this work, I used the NinaProDB1 dataset to reproduce results from EmgHandNet. Using a model with 42,893 parameters that leverages a hybrid ANN and SNN architecture, I achieved an overall test accuracy of **90.98%** (F1-score: 0.9082) with an inference time of approximately **1.6 microseconds per sample** on an Nvidia RTX 4070 with 8GB VRAM.

## Methodology  

### Preprocessing  
- Data preprocessing was done in MATLAB. 
- Applied a 10th order high-pass filter at 15 Hz because EMG frequencies of interest lie between 20-500 Hz, and no content above 500 Hz was observed. 
- Extracted time-domain features per window (200 ms window with 50% overlap): RMS, MAV, VAR. 
- Extracted frequency-domain features: FFT power and Peak frequency. 

In Python: 
- Applied SMOTE to address class imbalance. 
- Split data into train/validation/test sets with an 80/10/10 ratio. 
- Reshaped features to add a time dimension for compatibility with the Leaky Integrate-and-Fire (LIF) layer.

### Model  
- The model includes two linear layers before feeding into a LIF layer and a final fully connected layer for classification. 
- This structure allows the LIF layer to process dense temporal data per time slice. 
- No encoding was used, as EMG signals inherently contain rich temporal features, and encoding degraded performance.

| Layer (type)           | Output Shape | Param # |
|-----------------------|--------------|---------|
| Linear-1              | [-1, 128]    | 6,528   |
| Linear-2              | [-1, 256]    | 33,024  |
| PiecewiseLeakyReLU-3  | [-1, 256]    | 0       |
| LIFNode-4             | [-1, 256]    | 0       |
| Linear-5              | [-1, 13]     | 3,341   |

**Total params:** 42,893  
**Trainable params:** 42,893  
**Non-trainable params:** 0  

**Input size (MB):** 0.00  
**Forward/backward pass size (MB):** 0.01  
**Params size (MB):** 0.16  
**Estimated Total Size (MB):** 0.17  



## Evaluation  
- Achieved a test accuracy of **90.89%** with fewer than 100K parameters. 
- This is competitive compared to baselines like EmgHandNet with DNNs (80-90%) and Atorzi et al. with SVMs (70-85%). 
- Detailed statistics, confusion matrices, and training logs including inference times are available in the logs directory.

## Notes  
- Future work will explore multimodal fusion for more robust learning using this architecture. 
- This model presents a promising low-power, low-compute, energy-efficient solution for real-time prosthetic control.

