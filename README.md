app link: https://house-price-model-deploy.streamlit.app/

# House Prices EDA Project

## Dataset
Ames Housing dataset (`house_prices.csv`) — 1460 rows, 81 columns.
Target: `SalePrice` — regression (numeric).

## Topic 1 & 2: Loading, First Look, Missing Values
- Loaded data, checked `.head()`, `.shape`, `.info()`, `.describe()`
- Found 19 columns with missing values
- Distinguished two types of missing data:
  - "Doesn't have this feature" (PoolQC, Alley, Fence, FireplaceQu, MiscFeature,
    MasVnrType, Garage*, Bsmt* categorical columns) → filled with 'None'
  - Genuinely missing (LotFrontage, MasVnrArea, GarageYrBlt, Electrical)
    → filled with median (numeric) or mode (categorical)
- Verified: `df.isnull().sum().sum()` = 0 after cleaning

## Skewness — Quick Concept
- Skewness = how lopsided a distribution is
- Right-skewed: long tail toward big values, mean > median (e.g. SalePrice)
- Fix for right-skew: log transform, `np.log1p(col)`

## Topic: Target Variable Analysis (SalePrice)
- `SalePrice` is right-skewed (skew = 1.88, mean > median)
- Applied `np.log1p` transform, saved as `SalePrice_log` (skew = 0.12, roughly symmetric)
- Keeping the original `SalePrice` column as well for reference

- Scatter plot of GrLivArea vs SalePrice showed 2 outliers (GrLivArea > 4000, SalePrice < 300000)
- Dropped both rows (1460 → 1458 rows) since they don't follow the general trend

## Topic: Bivariate Analysis (Regression)
- Top correlations with SalePrice: OverallQual (0.79), GrLivArea (0.71), GarageCars (0.64), GarageArea (~0.62)
- Found multicollinearity: GarageCars vs GarageArea (0.89), TotalBsmtSF vs 1stFlrSF (0.8)
- Both pairs measure nearly the same thing, so only one of each would be kept when modeling
- Scatter plot of GrLivArea vs SalePrice showed 2 outliers (large area, low price); dropped them (1460 → 1458 rows)

- Neighborhood strongly affects price: median SalePrice ranges from ~$103,000 (IDOTRR) to ~$315,000 (NridgHt)
- Most expensive: NridgHt, NoRidge, StoneBr. Cheapest: IDOTRR, BrDale (and likely MeadowV)
- Categorical columns need groupby/boxplots, since the correlation heatmap covers numeric columns only

## Summary of Findings (House Prices)
- **Missing values:** most were not truly missing. NaN meant the house lacks that feature (pool, alley, fence, garage, basement), so they were filled with 'None'. Only LotFrontage, MasVnrArea, GarageYrBlt and Electrical were genuinely missing (median/mode fill).
- **Target:** SalePrice was right-skewed (1.88). Applied log(1+x) → skew 0.12.
- **Outliers:** 2 houses with very large GrLivArea but low price were dropped (1460 → 1458 rows).
- **Strongest predictors:** OverallQual (0.79), GrLivArea (0.71), GarageCars (0.64), plus Neighborhood (median price from ~$103k to ~$315k).
- **Multicollinearity:** GarageCars/GarageArea (0.89) and TotalBsmtSF/1stFlrSF (0.8) are near-duplicates, so only one of each would be kept when modeling.

SalePrice was right-skewed (skew 1.88), so the model was trained on log1p(SalePrice) (skew 0.12). Predictions were converted back to dollars with expm1 before computing MAE and RMSE."



















# House Prices: EDA and Model Training

Exploratory data analysis and model training on the Ames Housing dataset. The project predicts a house's **sale price** with **linear regression**, and also predicts whether a house is **expensive or not** with **logistic regression**. The results are deployed as a Streamlit app.

## Contents
1. [Dataset](#1-dataset)
2. [EDA](#2-eda-exploratory-data-analysis)
3. [Model Training Pipeline](#3-model-training-pipeline)
4. [Linear Regression](#4-linear-regression-predicting-the-price)
5. [Logistic Regression](#5-logistic-regression-expensive-or-not)
6. [Model Comparison](#6-model-comparison)
7. [Streamlit App](#7-streamlit-app)
8. [Key Concepts Summary](#8-key-concepts-summary)
9. [Key Learnings](#9-key-learnings)
10. [How to Run](#10-how-to-run)

---

## 1. Dataset

- **Source:** Ames Housing dataset (`house_prices.csv`)
- **Size:** 1460 rows, 81 columns (1458 rows after removing 2 outliers)
- **Target:** `SalePrice` (a number, so this is a **regression** problem)
- **Feature types:** numeric (e.g. `GrLivArea`, `LotArea`, `YearBuilt`) and categorical text columns (e.g. `Neighborhood`, `KitchenQual`, `GarageType`)

---

## 2. EDA (Exploratory Data Analysis)

EDA means understanding the data **before** modelling. The goal is to find problems (missing values, skew, outliers) and patterns (which columns matter).

### 2.1 Loading and first look
- Checked `.head()`, `.shape`, `.info()` and `.describe()`.
- Found **19 columns with missing values**.

### 2.2 Missing values

Two different kinds of "missing" were found, and they need different treatment:

| Type | Columns | Meaning | Fix |
|---|---|---|---|
| Structural (not really missing) | PoolQC, Alley, Fence, FireplaceQu, MiscFeature, MasVnrType, Garage* and Bsmt* categorical columns | NaN means the house **doesn't have** that feature | fill with `'None'` |
| Genuinely missing | LotFrontage, MasVnrArea, GarageYrBlt, Electrical | the value exists but wasn't recorded | median (numeric) / mode (categorical) |

After cleaning, `df.isnull().sum().sum()` was 0.

**Concept:** filling every NaN with the median would have been wrong for a column like `PoolQC`, because most houses simply have no pool.

### 2.3 Skewness: Quick Concept
- Skewness = how lopsided a distribution is
- Right-skewed: long tail toward big values, mean > median (e.g. SalePrice)
- Fix for right-skew: log transform, `np.log1p(col)`

### 2.4 Target variable analysis (SalePrice)
- `SalePrice` is **right-skewed** (skew = 1.88, mean > median): most houses are moderately priced, and a few very expensive ones stretch the tail.
- Applied `np.log1p`, saved as `SalePrice_log`: skew became **0.12**, roughly symmetric.
- The original `SalePrice` column was kept for reference.

### 2.5 Outliers
- A scatter plot of `GrLivArea` vs `SalePrice` showed **2 outliers** (`GrLivArea` > 4000 but `SalePrice` < 300000).
- They don't follow the general trend (huge house, low price), so they were dropped (1460 to 1458 rows).

**Concept:** an outlier can pull a regression line towards it and distort the model. These two were unusual sales that don't represent normal houses.

### 2.6 Bivariate analysis
- **Top correlations with SalePrice:** OverallQual (0.79), GrLivArea (0.71), GarageCars (0.64), GarageArea (about 0.62).
- **Neighborhood** strongly affects price: median `SalePrice` ranges from about $103,000 (IDOTRR) to about $315,000 (NridgHt). Most expensive: NridgHt, NoRidge, StoneBr. Cheapest: IDOTRR, BrDale.
- Categorical columns need `groupby` / boxplots, because the correlation heatmap covers numeric columns only.

### 2.7 Multicollinearity
- GarageCars vs GarageArea: **0.89**. TotalBsmtSF vs 1stFlrSF: **0.8**.
- Each pair measures nearly the same thing, so only one of each is kept when modelling.

**Concept:** when two features carry almost the same information, a linear model can't tell which one deserves the credit, so the weights become unstable and hard to interpret.

### 2.8 EDA summary

| Finding | Decision for modelling |
|---|---|
| Structural NaNs | fill with `'None'` |
| Genuine NaNs | median / mode |
| SalePrice skew 1.88 | train on `log1p(SalePrice)` |
| 2 outliers | drop them |
| Correlated pairs | drop `GarageArea` and `1stFlrSF` |
| Strong predictors | OverallQual, GrLivArea, GarageCars, Neighborhood |

---

## 3. Model Training Pipeline

The model notebook starts from the raw CSV and repeats the EDA cleaning, then adds the steps only needed for modelling.

| Step | What was done | Why |
|---|---|---|
| 1. Cleaning | drop outliers and `Id`, fill structural NaNs with `'None'` | same decisions as EDA; `Id` is just a row number |
| 2. Target | `y = np.log1p(SalePrice)` | fixes right skew |
| 3. Features | drop `SalePrice`, `GarageArea`, `1stFlrSF` | remove the target and the duplicated information |
| 4. Encoding | `pd.get_dummies(X, drop_first=True, dtype=int)` | models need numbers, not text |
| 5. Split | `train_test_split(test_size=0.2, random_state=42)` | 1166 train / 292 test houses |
| 6. Fill NaN | numeric NaNs filled with the **train** median | no information from test leaks in |
| 7. Scaling | `StandardScaler`, fitted on train only | features on a similar scale |
| 8. Training | linear regression and logistic regression | see sections 4 and 5 |

### Concepts used in the pipeline

**One-hot encoding (`get_dummies`).** A text column like `Neighborhood` becomes one 0/1 column per category (`Neighborhood_NridgHt`, `Neighborhood_IDOTRR`, ...). The feature count grows from about 77 to about 200 (print `X.shape` to see your exact number).

**`drop_first=True`.** A column with n categories needs only n - 1 dummy columns, because if all the others are 0, the dropped category is implied. This avoids redundant columns that confuse linear models.

**Train/test split.** The model learns on the training set (80%) and is judged on the test set (20%), which it has never seen. This gives an honest measure of performance.

**`random_state=42`.** Fixes the random shuffle, so the same houses land in train and test every time. This makes results reproducible and the two models comparable on the same test houses. The number 42 itself has no special meaning.

**Data leakage.** Any value learned from the test set (a median, a mean, a threshold) would give the model a hint it shouldn't have. That's why the medians, the scaler and the "expensive" threshold are all computed on the **train set only**. (Encoding doesn't learn anything from the data, so it is safe before the split.)

**Feature scaling (`StandardScaler`).** Rescales each column to mean 0 and standard deviation 1, so a column in the thousands (`LotArea`) doesn't dominate one in single digits (`OverallQual`). `fit_transform` on train learns the mean and std; `transform` on test reuses the train values.

**Log transform and `expm1`.** The model learns on log prices. Predictions are converted back to dollars with `np.expm1` (the exact reverse of `log1p`) before computing MAE and RMSE.

---

## 4. Linear Regression (predicting the price)

### Idea
Linear regression draws the best-fit line (really a plane in many dimensions) through the data:

`log(price) = w1*x1 + w2*x2 + ... + wn*xn + b`

- `w` (weights): how strongly each feature pushes the price up or down
- `b` (bias): the starting value when all features are 0

### How it learns
It chooses the weights that minimise the **Mean Squared Error (MSE)**, the average of (actual - predicted)² over the training houses. Squaring makes every error positive and punishes big errors more. This can be solved directly with a formula or step by step with **gradient descent** (nudging the weights in the direction that reduces the error).

### Results (292 test houses)

| Metric | Value | Meaning |
|---|---|---|
| R² (price scale) | **0.914** | explains about 91% of the variation in prices |
| R² (log scale) | 0.890 | the same, measured on the log scale |
| MAE | **$15,586** | average size of the mistake |
| RMSE | **$21,827** | typical mistake, with big misses weighted more |

### Metrics explained

- **MAE** (Mean Absolute Error) = average of |actual - predicted|. Easy to read: the model is off by about $15.6k on average, roughly 8-9% of a typical house price.
- **RMSE** (Root Mean Squared Error) = square root of the average squared error. Because errors are squared, big misses count more. RMSE ($21.8k) being higher than MAE ($15.6k) means most predictions are close, but a few houses (likely expensive ones) are off by much more.
- **R²** compares the model with the laziest model, which predicts the average price for every house. R² = 1 is perfect and R² = 0 is no better than the average. Here 0.914 means the model removes about 91% of that lazy model's error.
- R² on the price scale is slightly higher than on the log scale because expensive houses (which the model predicts well) dominate the variance in dollars. Always state which scale you used.

---

## 5. Logistic Regression (expensive or not)

### Why a new target is needed
Logistic regression predicts **categories**, but `SalePrice` is a number. So a yes/no target was created:

- `expensive = 1` if the price is above the **median training price**, otherwise `0`
- The median comes from the train set only (no leakage), and it splits the training houses roughly 50/50.

The same split and scaled features from the linear model were reused, so both models face the same test houses.

### Idea
1. **Linear score:** `z = w1*x1 + ... + b`
2. **Sigmoid:** `p = 1 / (1 + e^(-z))` squashes z into a probability between 0 and 1.
3. **Threshold:** `p >= 0.5` means predict "expensive", otherwise "not expensive".

The weights are learned by minimising **log loss** (cross-entropy): confident wrong predictions are punished heavily, confident right ones are rewarded. The minimisation uses gradient descent.

### Results (292 test houses)

**Accuracy: 0.921** (269 of 292 correct)

| | Predicted: not expensive | Predicted: expensive |
|---|---|---|
| **Actual: not expensive** | 130 (TN) | 11 (FP) |
| **Actual: expensive** | 12 (FN) | 139 (TP) |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 (not expensive) | 0.92 | 0.92 | 0.92 | 141 |
| 1 (expensive) | 0.93 | 0.92 | 0.92 | 151 |

### Metrics explained

- **Confusion matrix:** rows are the actual class, columns are the predicted class. The diagonal (130 and 139) holds the correct predictions.
  - **False positive (11):** a not-expensive house called expensive.
  - **False negative (12):** an expensive house called not expensive.
- **Precision (class 1) = TP / (TP + FP) = 139 / 150 = 0.93.** When the model says "expensive", it's right 93% of the time.
- **Recall (class 1) = TP / (TP + FN) = 139 / 151 = 0.92.** Of all truly expensive houses, the model found 92%.
- **F1 = 2 * (precision * recall) / (precision + recall).** One score that is high only when both are high.
- **Support:** the number of real houses of each class in the test set.
- Since the classes are almost balanced and neither mistake is clearly worse, accuracy is a fair summary here. The mistakes most likely sit near the median price, where houses are genuinely borderline.

---

## 6. Model Comparison

Linear regression's predicted prices were also converted into "expensive / not expensive" (price above the median or not), so both models are scored on the same question and the same 292 houses.

| Model | Task | Result |
|---|---|---|
| Linear regression | predict price | R² 0.914, MAE about $15.6k, RMSE about $21.8k |
| Logistic regression | expensive or not | accuracy 0.921 |
| Linear regression (price cut at median) | expensive or not | accuracy 0.928 |

**Takeaway:** the two models differ by only 2 houses out of 292, so they perform about equally on the yes/no question, and a different `random_state` could flip the order. Linear regression does well because it learned from the actual prices (more detail than 0/1 labels). For the price itself, only linear regression gives a number, and logistic regression is the natural choice when the target is a category.

---

## 7. Streamlit App

The app has three tabs:

- **Predict:** enter the 10 main house details and get an estimated price, the typical error (plus or minus MAE), and the probability that the house is above the median price.
- **Model Performance:** R², MAE and RMSE for linear regression, plus the accuracy, classification report and confusion matrix for logistic regression, and the comparison table.
- **Dataset:** a preview of the data, summary statistics, the SalePrice distribution and median price by neighborhood.

**How the prediction works:** a house has about 200 model features, so the app asks for only 10 and fills every other feature with the **median (typical) house** from the training data. The result is an approximate estimate. The saved model, scaler, feature column list and medians make sure the input goes through exactly the same steps as in training.

**Live app:** *(add link)*

---

## 8. Key Concepts Summary

| Concept | One-line meaning |
|---|---|
| EDA | understanding the data before modelling |
| Structural missing values | NaN meaning "doesn't have this feature" |
| Skewness | how lopsided a distribution is |
| Log transform | `log1p` reduces right skew; `expm1` reverses it |
| Outlier | a point that doesn't follow the general trend |
| Correlation | how strongly two numeric columns move together |
| Multicollinearity | two features carrying nearly the same information |
| One-hot encoding | text categories turned into 0/1 columns |
| `drop_first=True` | removes one redundant dummy column |
| Train/test split | learn on one part, judge on a hidden part |
| `random_state` | fixes the shuffle for reproducibility |
| Data leakage | test information accidentally used during training |
| Feature scaling | puts columns on a similar scale |
| Linear regression | best-fit line predicting a number |
| MSE | average of squared errors; what linear regression minimises |
| MAE / RMSE | average mistake / mistake that punishes big errors |
| R² | share of price variation the model explains |
| Logistic regression | linear score plus sigmoid, predicting a category |
| Sigmoid | converts any score into a probability between 0 and 1 |
| Log loss | error measure that punishes confident wrong answers |
| Gradient descent | step-by-step weight updates that reduce the error |
| Confusion matrix | table of correct and wrong predictions per class |
| Precision / recall / F1 | false-alarm rate / missed-case rate / balance of both |

---

## 9. Key Learnings

- **EDA decisions carry into modelling:** the skew led to the log transform, the outliers were removed, and the correlated pairs were reduced.
- **Not every NaN is missing data.** Check what it means before filling it.
- **Match the metric to the problem:** R², MAE and RMSE for numbers; accuracy, precision, recall and F1 for classes.
- **Avoid leakage:** compute medians, scaling and thresholds from the training set only.
- **Evaluate on unseen data,** and remember a single split carries some luck: a 2-house difference is not a real winner.
- **Choose the model to fit the target:** linear regression for numbers, logistic regression for categories.

---

## 10. How to Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

**Files**
- `eda_project.ipynb`: EDA
- `house_price_model.ipynb`: model training (linear and logistic regression)
- `app.py`: Streamlit app
- `house_prices.csv`: dataset
- `*.pkl`, `house_metrics.json`: saved models, scaler, feature list, medians and metrics used by the app

**Tech stack:** Python, pandas, NumPy, scikit-learn, matplotlib / seaborn, Streamlit, joblib.