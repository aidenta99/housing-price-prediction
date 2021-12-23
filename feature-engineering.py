from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.options.display.max_rows = 1000

data_dir = Path("data/")
img_dir = Path("../img")

# Load data
df = pd.read_csv(data_dir/"housing-data.csv", index_col="Order")

# Check which column contains missing values 
df.isnull().sum()
# Alley              2459
# Fireplace Qu       1292
# Pool QC            2627
# Fence              2118
# Misc Feature       2543

df['Utilities'].value_counts()
df['Street'].value_counts()
# ----Utilities----
# AllPub    2635
# NoSewr       2
# ----Street----
# Pave    2627
# Grvl      10

# Remove the columns in which most values are null OR most rows fall into one value (e.g. Ultilities, Street)
df = df.copy()
df = df.drop(['PID', 'Alley', 'Utilities', 'Street', 'Pool QC', 'Fence', 'Misc Feature'], axis=1)

# Some of the non-numeric df are stored as numbers, we convert them into strings 
df['MS SubClass'] = df['MS SubClass'].apply(str)
df['Yr Sold'] = df['Yr Sold'].astype(str)
df['Mo Sold'] = df['Mo Sold'].astype(str)

# Handle missing values in other columns

# For df which has a value of 'typical'/'average', we impute nulls into that value
df['Functional'] = df['Functional'].fillna('Typ')
df['Electrical'] = df['Electrical'].fillna("SBrkr")
df['Kitchen Qual'] = df['Kitchen Qual'].fillna("TA")
df["PoolQC"] = df["PoolQC"].fillna("None")
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0]) 
df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    df[col] = df[col].fillna(0)

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    df[col] = df[col].fillna('None')

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df[col] = df[col].fillna('None')

df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

# For Zoning, we fill in nulls based on MS SubClass that it belongs to  
df['MS Zoning'] = df.groupby('MS SubClass')['MS Zoning'].transform(lambda x: x.fillna(x.mode()[0]))
# For Lot Frontage, we fill in nulls based on Neighborhood that it belongs to 
df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(lambda x: x.fillna(x.median()))

# Fill in the rest of the nulls
categorical = []
for i in df.columns:
    if df[i].dtype == object:
        categorical.append(i)
df.update(df[categorical].fillna('None'))

numerical = []
for i in df.columns:
    if df[i].dtype in ['int64', 'float64']:
        numerical.append(i)
df.update(df[numerical].fillna(0))

# Apply one-hot encoding into the df:  
final_df = pd.get_dummies(df).reset_index(drop=True)

# Feature engineering:

# Add numerical columns
df['YrBltAndRemod']=df['YearBuilt']+df['YearRemodAdd']
df['TotalSF']=df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['Total_Sqrt_Footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF'])
df['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
df['Total_Porch_SF'] = (df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])

# Add binary df
df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)