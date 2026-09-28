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