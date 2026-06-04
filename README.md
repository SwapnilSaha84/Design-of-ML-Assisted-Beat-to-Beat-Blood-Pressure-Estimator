# Design-of-ML-Assisted-Beat-to-Beat-Blood-Pressure-Estimator
# Developed an end-to-end system for continuous, non-invasive blood pressure monitoring using PPG signals, machine learning, and embedded hardware.
# Utilized the MIMIC-III Waveform Database for model training and validation across diverse physiological conditions.
# Preprocessed PPG signals using a 4th-order Butterworth filter to remove noise, motion artifacts, and baseline drift.
# Extracted meaningful signal features using a Variational Autoencoder (VAE) for efficient representation learning.
# Implemented a Random Forest Regression model to estimate Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP).
# Evaluated model performance using Mean Absolute Error (MAE) and Standard Deviation of Error (SDE).
# Designed a hardware prototype using a TCRT1000 PPG sensor, analog front-end, ESP32, and Raspberry Pi 4.
# Developed a real-time physiological signal acquisition and processing pipeline with on-device inference.
# Displayed heart rate (BPM) on an I2C 16×2 LCD for local monitoring.
# Demonstrated a scalable and cost-effective solution for wearable healthcare and remote patient monitoring applications.
