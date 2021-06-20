from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.options.display.max_rows = 1000

data_dir = Path("data/")
img_dir = Path("../img")

# Load data
all_data = pd.read_csv(data_dir / "housing-data.csv", index_col="Order")

# Check which column contains missing values 
all_data.isnull().sum()
# Alley              2459
# Fireplace Qu       1292
# Pool QC            2627
# Fence              2118
# Misc Feature       2543

all_data['Utilities'].value_counts()
all_data['Street'].value_counts()
# ----Utilities----
# AllPub    2635
# NoSewr       2
# ----Street----
# Pave    2627
# Grvl      10

# Remove the columns in which most values are null OR most rows fall into one value (e.g. Ultilities, Street)
features = all_data.copy()
features = features.drop(['PID', 'Alley', 'Utilities', 'Street', 'Pool QC', 'Fence', 'Misc Feature'], axis=1)

# Some of the non-numeric features are stored as numbers, we convert them into strings 
features['MS SubClass'] = features['MS SubClass'].apply(str)
features['Yr Sold'] = features['Yr Sold'].astype(str)
features['Mo Sold'] = features['Mo Sold'].astype(str)

# Handle missing values in other columns

# For features which has a value of 'typical'/'average', we impute nulls into that value
features['Functional'] = features['Functional'].fillna('Typ')
features['Electrical'] = features['Electrical'].fillna("SBrkr")
features['Kitchen Qual'] = features['Kitchen Qual'].fillna("TA")

# For Zoning, we fill in nulls based on MS SubClass that it belongs to  
features['MS Zoning'] = features.groupby('MS SubClass')['MS Zoning'].transform(lambda x: x.fillna(x.mode()[0]))
# For Lot Frontage, we fill in nulls based on Neighborhood that it belongs to 
features['Lot Frontage'] = features.groupby('Neighborhood')['Lot Frontage'].transform(lambda x: x.fillna(x.median()))

# Fill in the rest of the nulls
categorical = []
for i in features.columns:
    if features[i].dtype == object:
        categorical.append(i)
features.update(all_data[categorical].fillna('None'))

numerical = []
for i in features.columns:
    if features[i].dtype in ['int64', 'float64']:
        numerical.append(i)
features.update(features[numerical].fillna(0))

# Apply one-hot encoding into the features:  
final_features = pd.get_dummies(features).reset_index(drop=True)