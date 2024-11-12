# Automated EEG Signal Classification for Neurocritical Care: A Machine Learning Approach

## Overview

This project aims to develop an automated machine learning pipeline to classify EEG signals for neurocritical care applications. The notebook is built for a Kaggle competition sponsored by the Sunstella Foundation, Persyst, Jazz Pharmaceuticals, and the Clinical Data Animation Center (CDAC). The goal is to create a model that detects and classifies various brain activity patterns, including seizures, using EEG data. This competition underscores the potential of AI-driven EEG analysis to significantly improve the speed, accuracy, and cost-effectiveness of neurocritical care diagnostics.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Processing](#data-processing)
3. [Model Design](#model-design)
4. [Training and Evaluation](#training-and-evaluation)
5. [Results](#results)
6. [Conclusion](#conclusion)
7. [Usage](#usage)
8. [Requirements](#requirements)

## Introduction

Manual analysis of EEG data by specialists is time-consuming, costly, and prone to variability in interpretation. This notebook introduces a machine learning model based on the EfficientNetV2 architecture that aims to automate the classification of EEG signals. By utilizing advanced techniques such as stratified group K-fold cross-validation, parallel processing, and augmentation, this project provides a robust pipeline for handling EEG data and training a deep learning model to classify seizure-related events accurately.

## Data Processing

The EEG data is processed from its raw format into a structured and analysis-friendly representation using the following steps:

- **File Paths Setup**: Paths for spectrogram files are structured for easy access.
- **Parallel Processing of Spectrograms**: Using joblib, the notebook reads and transforms spectrograms in parallel, optimizing processing time.
- **Data Augmentation**: Functions to augment EEG spectrograms using keras_cv for advanced masking and MixUp augmentation.
- **Dataset Splitting**: A Stratified Group K-Fold Cross-Validation strategy is employed to prevent data leakage, ensuring records from the same patient appear in either the training or validation set exclusively.

## Model Design

The model uses an EfficientNetV2 architecture, optimized with the Kullback-Leibler Divergence (KL Divergence) loss function. This approach is chosen due to its efficiency and capability to learn complex patterns in EEG data. Key components include:

- **Architecture**: EfficientNetV2 is adapted for this task with six output classes.
- **Loss Function**: KL Divergence is applied to encourage predictions that closely resemble observed probability distributions.
- **Learning Rate Scheduling**: A cosine learning rate decay strategy, implemented with callbacks, helps optimize training.

## Training and Evaluation

The model is trained for 13 epochs with a batch size of 64. The training process includes:

- **Callbacks**: Learning rate scheduling and checkpoint saving callbacks.
- **Evaluation Metrics**: Validation loss is tracked throughout training to identify the best-performing model.

## Results

The model achieves a validation loss of 0.8329 on the dataset, indicating promising performance in classifying EEG patterns. The trained model is saved, and predictions are generated for the test dataset, with results saved in `submission.csv`.

## Conclusion

This project demonstrates the potential of machine learning to enhance neurocritical care by automating EEG analysis. The efficient processing and deep learning techniques applied here are steps towards accurate and rapid classification of brain activity. Future directions may include hyperparameter tuning and exploring additional architectures for improved accuracy.

## Usage

To run this notebook on Kaggle, ensure you have access to the competition dataset. Follow these steps:

1. Set up the environment by loading the required libraries.
2. Run each code block sequentially, starting with data processing.
3. Train the model and observe training metrics.
4. Generate predictions for the test set by running the final code blocks.

## Requirements

- `tensorflow` >= 2.15.0
- `keras` >= 3.0.5
- `keras_cv` >= 0.8.2
- `pandas`, `numpy`, `cv2`, `joblib`, `glob`, `tqdm`
- Compatible GPU for accelerated training (CUDA setup recommended)

## Acknowledgements

This project is part of a Kaggle competition hosted by the Sunstella Foundation in collaboration with Persyst, Jazz Pharmaceuticals, and the Clinical Data Animation Center (CDAC). We thank them for providing the data and platform to advance neurocritical care research.

For additional information, consult the full documentation within each code block in the notebook.
