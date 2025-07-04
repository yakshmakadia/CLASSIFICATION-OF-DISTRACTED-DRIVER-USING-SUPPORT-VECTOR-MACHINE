# Distracted Driver Classification

## Overview
This project focuses on classifying distracted driver behavior using a combination of Convolutional Neural Networks (CNN), Principal Component Analysis (PCA), and Support Vector Machines (SVM). By leveraging deep learning for feature extraction and machine learning for classification, this system aims to improve road safety by identifying distracted driving behaviors in real-time.

## Features
- **Convolutional Neural Networks (CNN)**: Used for feature extraction from images.
- **Principal Component Analysis (PCA)**: Reduces feature dimensionality for faster processing.
- **Support Vector Machines (SVM)**: Classifies driver behavior efficiently.
- **Preprocessing Pipeline**: Includes image resizing, normalization, and augmentation.
- **Dataset Compatibility**: Built to work with the Kaggle *State Farm Distracted Driver Detection* dataset.

## Installation
Ensure you have Python 3.x installed, then install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. **Prepare the dataset**: Download and place it in the appropriate directory.
2. **Run the main script**:
```bash
python main.py
```
3. **Evaluate performance**: The script will output accuracy and other evaluation metrics.

## Code Structure
- `main.py`: Core script to execute preprocessing, feature extraction, PCA, and classification.
- `requirements.txt`: Lists necessary dependencies.
- `config.yaml`: Configuration file for model parameters.
- `LICENSE`: MIT License for open-source distribution.

## Dataset Details
This project utilizes the *State Farm Distracted Driver Detection* dataset, which contains labeled images of drivers engaged in various activities such as:
- Safe driving
- Texting while driving
- Talking on the phone
- Eating, drinking, and other distractions

## Performance
By implementing PCA, training time was reduced significantly while maintaining high accuracy (approximately 99%).

## Future Enhancements
- Implement real-time detection using OpenCV.
- Deploy as a web or mobile application.
- Explore deep learning alternatives such as Vision Transformers.
