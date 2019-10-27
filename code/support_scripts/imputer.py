# Missing Forest imputation attempt

# Import dependencies
import numpy as np
import pandas as pd
from missingpy import MissForest

# Load data
train = pd.read_csv("/home/nishant/Desktop/IDA Project/mod_data/train.csv")
cols = train.columns.tolist()

# Impute values
# Function returns a numpy ndarray, which we convert to DataFrame again
imputer = MissForest()

print("[INFO] Imputation started")
X_imputed = imputer.fit_transform(train.values)

print("[INFO] Imputation complete")
train_mf = pd.DataFrame(X_imputed, columns=cols)

# Save new DataFrame to drive
train_mf.to_csv("/home/nishant/Desktop/IDA Project/mod_data/train_mf.csv", index=False)