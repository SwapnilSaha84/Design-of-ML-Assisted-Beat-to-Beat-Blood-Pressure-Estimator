# Design-of-ML-Assisted-Beat-to-Beat-Blood-Pressure-Estimator
Overview

This project presents an end-to-end system for continuous, non-invasive blood pressure monitoring by integrating biomedical signal processing, machine learning, and embedded hardware. The system estimates Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP) from Photoplethysmography (PPG) signals and supports real-time physiological monitoring. Real-time acquisition and processing of physiological signals and continuous cardiovascular monitoring with on-device inference.

Key Features

1)Non-invasive blood pressure estimation using PPG signals.

2)Signal preprocessing using a 4th-order Butterworth filter.

3)Feature extraction using a Variational Autoencoder (VAE).

4)Blood pressure prediction using Random Forest Regression.
5)Real-time signal acquisition using ESP32 and Raspberry Pi.

6)Local display of physiological parameters on an I2C 16×2 LCD.

7)Cost-effective and scalable healthcare monitoring solution.

Methodology

1)Acquired and validated data using the MIMIC-III Waveform Database.

2)Filtered PPG signals to remove noise, motion artifacts, and baseline drift.

3)Extracted latent features using a Variational Autoencoder (VAE).

4)Trained a Random Forest Regression model to estimate SBP and DBP.

5)Evaluated performance using Mean Absolute Error (MAE) and Standard Deviation of Error (SDE).

6)Implemented real-time deployment on Raspberry Pi 4 Model B.

Hardware Components

1)TCRT1000 Reflective PPG Sensor

2)Analog Front-End (TIA + High-Pass and Low-Pass Filters)

3)ESP32 Microcontroller

4)Raspberry Pi 4 Model B

5)I2C 16×2 LCD Display

Software & Tools

1)Python

2)Machine Learning (Random Forest, VAE)

3)Signal Processing

4)Raspberry Pi OS

5)ESP32 Firmware Development
