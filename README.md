# Phishing Email Detector

This project builds a phishing email classification model using a fine-tuned BERT transformer. It classifies email text as either "Phishing" or "Legit." The model was trained on 15,000 real emails and evaluated on a separate test set with high accuracy. A simple Gradio web app is included for live prediction.

## Overview

- Fine-tuned BERT (`bert-base-uncased`) for binary text classification
- Preprocessing includes case normalization and minimal cleaning to preserve phishing indicators
- Handled class imbalance through oversampling of phishing emails
- Achieved over 99% accuracy, precision, recall, and F1-score on test data
- Gradio interface for testing individual email predictions

## How to Run

1. Clone the repository:
   git clone https://github.com/rmel28/phishing-email-detector.git
   cd phishing-email-detector
Install dependencies:
pip install -r requirements.txt

python app.py
(Optional) To retrain the model:

python phishing_detector.py
Model Performance
Evaluation on the test set:

Accuracy: 99%

Precision (phishing): 0.99

Recall (phishing): 1.00

F1-score (phishing): 1.00

The confusion matrix showed very few misclassifications.

File Descriptions
fish.py: Full pipeline — data loading, preprocessing, training, and evaluation.
model/: Contains the saved BERT model and tokenizer after training.
phishing_email.csv.zip: Dataset used to train and test the model.

Requirements
To run this project, install the following libraries:
transformers==4.36.2
torch
scikit-learn
pandas
matplotlib
seaborn
nltk
datasets
gradio
wordcloud

You may also need to download NLTK stopwords:
import nltk
nltk.download('stopwords')
