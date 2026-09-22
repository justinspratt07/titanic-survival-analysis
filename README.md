# Titanic Passenger Survival Analysis

This project combines unsupervised and supervised learning to examine Titanic
passenger data. K-Means clustering identifies passenger groupings, while
Logistic Regression estimates passenger survival.

## Analysis workflow

### K-Means clustering

- Impute missing numeric and categorical values.
- Standardize numeric features and one-hot encode categorical features.
- Compare cluster counts from 2 through 6 using silhouette scores.
- Summarize the selected clusters and visualize them with PCA.

### Logistic Regression

- Use a stratified 80/20 train/test split.
- Build preprocessing and classification steps in one reproducible pipeline.
- Evaluate predictions with a classification report, confusion matrix, and
  ROC-AUC curve.
- Review the largest model coefficients for interpretability.

## Repository structure

```text
notebooks/
  titanic_analysis.ipynb
src/
  titanic_clustering.py
  titanic_classification.py
requirements.txt
```

## Dataset

The analysis expects a Titanic passenger spreadsheet named
`CS379T-Week-1-IP.xls` in the repository root. The spreadsheet used for the
coursework is not included because its redistribution terms were not supplied.
An equivalent dataset may be used if it contains these fields:

`pclass`, `age`, `sibsp`, `parch`, `fare`, `sex`, `embarked`, and `survived`.

## Setup

```bash
python -m venv .venv
python -m pip install -r requirements.txt
jupyter notebook notebooks/titanic_analysis.ipynb
```

## Outputs

- Silhouette-score chart
- PCA cluster visualization
- Cluster profile CSV
- Confusion matrix
- ROC curve and ROC-AUC score
- Classification report and top Logistic Regression coefficients

## Author

[Justin Spratt](https://github.com/justinspratt07)

## Verified academic results

The original CS379 spreadsheet was rerun on 21 September 2026. Logistic Regression produced ROC-AUC 0.867 and accuracy 0.809 on 262 held-out records. K-Means selected three clusters among k=2 through 6 (silhouette 0.335). These are historical benchmark results, not a deployment claim. The dataset remains excluded from version control.

Run scripts from the repository root: `python -c "import sys; sys.path.insert(0, 'src'); from titanic_classification import run; print(run())"`. Replace `titanic_classification` with `titanic_clustering` to generate clustering outputs.
