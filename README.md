# Iris Flower Classification

A Jupyter notebook that classifies Iris flowers as Setosa, Versicolor, or Virginica from sepal and petal measurements. It uses the Iris dataset bundled with scikit-learn, so no data download is needed.

## What the notebook covers

- Dataset checks: shape, data types, missing values, descriptive statistics, and class balance
- Training set pairplot and box plots for all four measurements
- Feature ranking and a discussion of the most useful measurements
- Stratified 80/20 train/test split
- Logistic Regression and K-Nearest Neighbours with scaling inside each pipeline
- Five-fold cross-validation, classification reports, and confusion matrices
- A sample species prediction

## Results

Both classifiers reached **95.83% mean training cross-validation accuracy** and **93.33% test accuracy** (28 of 30 test flowers). Logistic Regression was selected using the stated tie rule. Petal length and petal width were the most discriminative individual measurements for this training split. The test set is small, so these figures are an estimate for this setup rather than a general guarantee.

## Run locally

1. Install Python 3 and run `python -m pip install jupyterlab scikit-learn pandas numpy matplotlib seaborn`.
2. Run `python -m jupyter lab` and open [`Iris_Flower_Classification.ipynb`](Iris_Flower_Classification.ipynb).
3. In JupyterLab, select a Python kernel and choose **Kernel → Restart Kernel and Run All Cells**.

**Dataset:** [`sklearn.datasets.load_iris`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html). The split and cross-validation use `random_state=42` for reproducibility.
