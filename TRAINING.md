# Chest X-ray findings experiment

This adds a small binary image classifier alongside the existing restoration workflow. It predicts the dataset labels **normal** or **pneumonia**; it does not generate a radiology report or recognize the many other findings covered by datasets such as CheXpert.

## Data and split

The run uses the official 224-pixel `PneumoniaMNIST+` file from [MedMNIST+ on Zenodo](https://zenodo.org/records/10519652), derived from pediatric chest X-rays. Its published train, validation and test splits are kept intact. The test split is only used after model selection. The dataset is CC BY 4.0 and is explicitly not intended for clinical use.

Download the dataset into `data/pneumoniamnist_224.npz`, then verify it against the published MD5 `d6a3c71de1b945ea11211b03746c1fe1`. The training script records the file hash in the checkpoint.

## Reproduce

```powershell
python -m scripts.train_pneumonia --epochs 10 --seed 42
python -m scripts.evaluate_pneumonia
streamlit run app.py
```

The training code chooses the checkpoint with the lowest validation loss. Evaluation then reports accuracy, ROC AUC, average precision, sensitivity, specificity, precision, F1, Brier score, and the confusion matrix on the untouched test split. The threshold is fixed at 0.5; it is not selected on the test data.

## Observed run

With seed 42 and 10 epochs, the best validation loss was 0.1466 (epoch 8). On the held-out test split (624 images), the result was ROC AUC 0.950, accuracy 0.803, sensitivity 0.995, specificity 0.483, precision 0.762, F1 0.863, and Brier score 0.151 at threshold 0.5. Confusion matrix: 113 normal images correctly predicted normal, 121 normal images predicted pneumonia, 2 pneumonia images predicted normal, and 388 pneumonia images correctly predicted pneumonia. These results show a substantial false-positive tradeoff at this threshold. The test metrics are saved in `results/pneumonia_test_metrics.json` and shown in the app.

## Limits

This is a compact portfolio baseline, not a clinically validated model. Its source data are pediatric, and performance on adult images, different hospitals, scanners, image processing, or prevalence may differ substantially. MedMNIST is a standardized benchmark with preprocessed low-resolution images, so this experiment is not equivalent to training and external validation on CheXpert. The app labels the score as experimental and applies the model only to the original uploaded image, not a restored output.
