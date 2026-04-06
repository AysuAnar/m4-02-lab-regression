![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Regression

## Overview

Regression is one of the fundamental tasks in supervised learning — predicting a continuous numeric value from a set of input features. In practice, choosing the right regression model (and tuning it properly) can make the difference between a usable prediction system and one that misses the mark entirely.

In this lab, you'll work with the California Housing dataset to build and compare several regression models: ordinary linear regression, regularized variants (Ridge and Lasso), and Support Vector Regression. You'll evaluate each model using standard metrics and develop an intuition for when regularization helps and how different algorithms handle the same prediction task.

This is your chance to go beyond fitting a single model and start thinking like a practitioner — comparing approaches, interpreting results, and making informed decisions about which model to deploy.

## Learning Goals

By the end of this lab, you should be able to:

- Train a baseline linear regression model and evaluate it with MSE, RMSE, MAE, and R².
- Apply Ridge and Lasso regularization and analyze how alpha affects model coefficients.
- Fit Support Vector Regression with different kernels and compare performance.
- Build a structured model comparison and justify your model selection with evidence.

## Setup and Context

You'll work inside a Jupyter Notebook for this lab. All analysis, code, and written interpretations should live in a single notebook so that your reasoning is visible alongside the output.

This lab builds on today's lesson about regression algorithms. You'll use scikit-learn's regression estimators, matplotlib for visualization, and pandas for organizing your results.

## Requirements

### Fork and clone

1. Fork this repository to your own GitHub account.
2. Clone the fork to your local machine.
3. Navigate into the project directory.

### Python environment

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Getting Started

1. Create a new Jupyter Notebook called **`m4-02-regression.ipynb`**.
2. Start with an import cell:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_style("whitegrid")
```

3. Work through the tasks in order. Each task builds on the previous one.
4. Include markdown cells between code cells to explain your observations and reasoning.

## Tasks

### Task 1: Baseline Model

Start with the simplest approach — a plain linear regression.

1. Load the California Housing dataset:

```python
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target
```

2. Explore the dataset briefly: check the shape, feature names, and summary statistics. What is the target variable (median house value) distribution?
3. Split the data into training and test sets (80/20 split, `random_state=42`).
4. Fit a `LinearRegression` model on the training data.
5. Evaluate on the test set using **MSE**, **RMSE**, **MAE**, and **R²**. Print the results in a clear format.
6. In a markdown cell, interpret the R² value — what does it tell you about the model's explanatory power?

### Task 2: Regularized Regression

Explore how regularization changes the model's behavior.

1. Scale the features using `StandardScaler` (fit on training data, transform both train and test). Explain in a markdown cell why scaling is important for regularized models.
2. Fit **Ridge** regression for alpha values `[0.01, 0.1, 1, 10, 100]`. Record the test R² for each.
3. Fit **Lasso** regression for the same alpha values. Record the test R² for each.
4. Create a plot showing **coefficients vs. alpha** for both Ridge and Lasso (side by side or overlaid). What happens to the coefficients as alpha increases? Which features does Lasso zero out first?
5. Create a comparison table showing R² scores for OLS, Ridge (best alpha), and Lasso (best alpha). Which regularization approach works best here?

### Task 3: Support Vector Regression

Try a non-linear approach with SVR.

1. Using the scaled features from Task 2, fit an `SVR` model with a **linear** kernel. Report the test R².
2. Fit SVR with an **rbf** kernel. Try at least three values of `C` (e.g., 0.1, 1, 10) and two values of `epsilon` (e.g., 0.1, 0.2). Report the best combination.
3. Fit SVR with a **poly** kernel (degree 2 and 3). Compare with rbf.
4. In a markdown cell, discuss: How does SVR compare to the linear models? Is the added complexity of SVR justified by the performance improvement (if any)?

> **Note:** SVR can be slow on larger datasets. If training takes too long, consider using a random subsample (e.g., 5,000 rows) for the SVR experiments.

### Task 4: Model Comparison

Bring everything together in a final analysis.

1. Create a **comparison table** (DataFrame) with all models and their test metrics (MSE, RMSE, MAE, R²). Include: LinearRegression, best Ridge, best Lasso, and best SVR.
2. Create a **predicted vs. actual** scatter plot for your best-performing model. Add a diagonal reference line (perfect prediction). How well does the model track the true values?
3. Create a **residual plot** (residuals vs. predicted values) for the best model. Do you see any patterns? What would a random scatter indicate?
4. Write a markdown conclusion: Which model performs best and why? What are the trade-offs between model complexity and performance? If you were deploying one of these models, which would you choose?

## Submission

### What to submit

- `m4-02-regression.ipynb` — your completed notebook with all code, outputs, and markdown explanations.

### Definition of done (checklist)

- [ ] California Housing dataset is loaded and explored.
- [ ] Baseline LinearRegression is trained and evaluated with four metrics.
- [ ] Ridge and Lasso are trained across multiple alpha values with coefficient plots.
- [ ] SVR is trained with at least three kernel types and hyperparameter variations.
- [ ] A comparison table summarizes all models side by side.
- [ ] Predicted vs. actual and residual plots are included for the best model.
- [ ] Markdown cells explain reasoning and model selection throughout.
- [ ] The notebook runs top-to-bottom without errors (`Kernel → Restart & Run All`).

### How to submit (Git workflow)

```bash
git add .
git commit -m "lab: complete regression models comparison"
git push origin main
```

Then open a **Pull Request** on the original repository with a brief description of your work.
