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
Tips:a Describe limitations of the baseline. List and prioritise next steps for model
development.

# Tackling the problem

1. Dataset loader (x-val)
2. EDA
3. pytorch lighting + wandb for model
4. predictor
5. training notebook
6. inference notebook

## Stuff I did
1. torch dataset + dataloader
2. fp16
3. efficientnet
4. wandb
5. torch lightning

## Considerations
1. dev time + training time
2. Resource constraint with only a local 2080 super (8gb)
3. Didn't want to use TPU as it's a hassle to set-up and only available on kaggle platform
3. k fold cross validation not feasible due to training time
3. picked up torch lightning
4. Just a few experiments to show model evaluation