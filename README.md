# Emotion Detection with BERT

Project: Emotion Detection with BERT
Dataset: shreyaspullehf/emotion_dataset_100k (100,000 balanced sentences across 10 emotion categories).

Frameworks: PyTorch + Hugging Face Transformers.

## Approach

* Model: Fine-tuned the pre-trained bert-base-uncased model for text classification.

* Pipeline: * Encoded emotion labels (0-9).

* Split data into 70% Train / 15% Val / 15% Test.

* Tokenized inputs using a custom PyTorch Dataset (Max Length: 128).

* Architecture: BERT base + Dropout (0.1) + Linear layer (10 classes).

* Optimization: Used Mixed Precision (AMP) and Parallel Data Loading to reduce training time from 1 hour to 8 minutes per epoch.



## Key Assumptions
* Max Length (128): Sufficient to capture full context for short sentences.

* Efficiency: Pre-trained BERT only requires 2–3 epochs to achieve high accuracy.

* Evaluation: Used Weighted F1-Score to ensure balanced performance across all emotions.

## Results & Observations
* Speed: Batch size (32) + AMP solved the initial training bottlenecks.

* Learning: The model converged rapidly, with loss dropping from 0.28 to 0.03 within the first epoch.

* Performance: Achieved 95% Accuracy and a 0.95 F1-score.

* Precision: Identifying "embarrassment" was nearly perfect (1.00), while "sadness" was the most common point of confusion.
