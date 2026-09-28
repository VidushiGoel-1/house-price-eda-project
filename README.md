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