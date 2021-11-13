# Pneumothorax Segmentation

1. Access and download the dataset from the Kaggle competition for Pneumothorax detection:
https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview
2. Perform appropriate exploratory data analysis and visualisation.
Tips: Discuss fields and calculate descriptive statistics. Visualise segmentation masks.
Describe implications for training and evaluation.
3. Build a best-effort algorithm to detect a visual signal for Pneumothorax as per the
competition evaluation metric.
Tips: Train an off-the-shelf baseline architecture. Explain architecture choice, including
references to any third-party code, experiments or write-ups.
4. Generate appropriate graphs / results table to assess model performance.
Tips: Generate and briefly discuss utility of measures used to track and assess performance.
Explain how to decide when to stop training.
5. Make a baseline submission to the late submission pool.
Tips: Use late submission pool to validate and benchmark results. Discuss.
6. Discuss shortcomings and the improvements you would make to the dataset, evaluation
metrics and algorithm.
Tips: Describe limitations of the baseline. List and prioritise next steps for model
development.

# Tackling the problem

1. Dataset loader (x-val)
2. EDA
3. pytorch lighting + wandb for model
4. predictor
5. training notebook
6. inference notebook
