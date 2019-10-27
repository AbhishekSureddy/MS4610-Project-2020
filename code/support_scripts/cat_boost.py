import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Encoding library
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("/home/nishant/Desktop/IDA Project/mod_data/train.csv")

# Label encode
le = LabelEncoder()
train['card_type'] = le.fit_transform(train.card_type)
train[['card_type', 'location_id']] = train[['card_type', 'location_id']].astype('category')

# X and y
y = train.default_ind.values
X = train.drop(['application_key', 'default_ind'], axis=1).values

# Split into train and eval
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
X_train_df = pd.DataFrame(X_train)

# XGB
# categorical_fs = [c for c, col in enumerate(X_train_df.columns) if 'cat' in col]
model = CatBoostClassifier(max_depth=5,
                           learning_rate=0.01,
                           iterations=500,
                           eval_metric='AUC',
                           random_state=123)
                           # cat_features=categorical_fs)
model.fit(X_train, y_train)

# Prediction
preds = model.predict(X_val)

# Scoring
score = roc_auc_score(y_val, preds)
print("ROC AUC Score: {}".format(score))

