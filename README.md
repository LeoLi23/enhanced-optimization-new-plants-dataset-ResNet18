# ResNet-18 Based Plant Disease Recognition

This repository contains a machine learning model based on the ResNet-18 architecture, optimized for the task of recognizing and classifying various plant diseases. This model has undergone extensive hyperparameter tuning and optimization to improve its accuracy and reliability.

## Project Overview

The project utilizes a publicly available dataset for training and validation, ensuring a broad and diverse range of plant disease images. Through the application of random search for hyperparameter tuning, regularization techniques, and other advanced optimization methods, the model's performance has significantly improved, particularly in its ability to accurately identify plant diseases.

## Key Features

- **ResNet-18 Architecture:** Leveraging the power of Residual Networks for deep learning tasks.
- **Random Search Hyperparameter Tuning:** Systematically adjusted hyperparameters to find the optimal configuration.
- **Regularization Techniques:** Applied to prevent overfitting and improve model generalization.
- **Advanced Optimization Techniques:** Utilized to enhance model performance and accuracy.
- **Comprehensive Evaluation:** Using a separate test set to validate the model, ensuring unbiased assessment.

## Visualizations

The model's performance and its ability to classify plant diseases accurately are demonstrated through various visualizations:

- **Confusion Matrices:** Providing insights into the accuracy and misclassifications of the model.
- **Grad-CAM Visualizations:** Highlighting the regions in the images that the model focuses on for making its predictions.
- **ROC Curves:** Illustrating the diagnostic ability of the model across various classification thresholds.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3
- TensorFlow
- Keras
- Other dependencies listed in `requirements.txt`

### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
pip install -r requirements.txt
