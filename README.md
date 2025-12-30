# EMG Hand Gesture Classification

## Introduction

Surface electromyography (sEMG) is a non-invasive technique for measuring muscle electrical activity and is widely used in human–machine interfaces, prosthetics, and rehabilitation systems. Reliable sEMG systems require low-noise analog front-ends, high common-mode rejection, stable power delivery, and careful PCB layout to preserve signal integrity, particularly given the low-amplitude (µV–mV) nature of EMG signals.

This project focuses on the design, fabrication, and validation of a custom sEMG acquisition board, with the aim of capturing high-fidelity muscle signals suitable for real-time and offline analysis. Rather than relying on commercial development kits, a fully custom PCB was developed to explore the practical challenges of analog signal conditioning, grounding, filtering, and mixed-signal system integration.
---

# Hardware Development and Testing
A custom sEMG acquisition hardware board has been designed and fabricated to enable lightweight, low-latency, and high-fidelity signal capture for real-time applications.

![Custom sEMG Acquisition Board](https://github.com/user-attachments/assets/d60003c4-8d64-4608-a2c1-6201ea7448be) 
<img width="1143" height="810" alt="image" src="https://github.com/user-attachments/assets/199e6229-784d-4c7a-840a-f8d00d9c123f" /> 
<img width="824" height="902" alt="image" src="https://github.com/user-attachments/assets/2c02a8f3-a584-4c64-aaec-d83945aa1f71" /> 
<img width="1463" height="804" alt="image" src="https://github.com/user-attachments/assets/3f3eba94-c6f3-4420-9cd1-e35a9bde7136" />




### Key Components and BOM Highlights:

- **Instrumentation Amplifiers:** Texas Instruments INA333 (3 units) for high CMRR and low noise EMG signal amplification  
- **Microcontroller:** STM32F103C8T6 (ARM Cortex-M3) for signal processing and data acquisition  
- **Power Management:** Microchip MIC5504-3.3 LDO regulator for stable 3.3V supply  
- **ESD Protection:** STMicroelectronics USBLC6-2SC6 for USB interface protection  
- **Passive Components:** High-quality capacitors (e.g., TDK X5R, KEMET), resistors (Susumu, Yageo), and ferrite bead (Murata) for noise filtering and stable operation  
- **Connectors:** USB-C receptacle, 3.5mm audio jacks for signal output and power input  
- **Oscillator:** EPSON 16MHz crystal for clock generation

---

# Methodology

## Data Acquisition and Preprocessing

Preprocessing of raw EMG signals was performed using a MATLAB script. Key preprocessing steps include:

- **Filtering:** A **10th order Butterworth bandpass filter** was implemented with cutoff frequencies between **20 Hz and 450 Hz** to remove motion artifacts, baseline drift, and high-frequency noise outside the typical EMG frequency range.  
- **Segmentation:** Signals were segmented according to gesture labels with pauses between gesture executions to minimize overlap.

