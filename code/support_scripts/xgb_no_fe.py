import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Plotting library
import matplotlib.pyplot as plt

# Encoding library
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("/home/nishant/Desktop/IDA Project/mod_data/train.csv")

# Label encode
le = LabelEncoder()
train['card_type'] = le.fit_transform(train.card_type)

# X and y
y = train.default_ind.values
X = train.drop(['application_key', 'default_ind'], axis=1).values

# Split into train and eval
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)

# XGB
model = XGBClassifier(max_depth=4,
                      learning_rate=0.01,
                      n_estimators=200,
                      objective='binary:logistic',
                      scale_pos_weight=2,
                      random_state=123)
# model.fit(X_train, y_train)
#
# # Prediction
# preds = model.predict(X_val)
#
# # Scoring
# score = roc_auc_score(y_val, preds)
# print("ROC AUC Score: {}".format(score))

from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1, 5),
                                                         scoring='roc_auc', cv=2)

# Plot training and validation curve
plt.figure(figsize=(6,6))
plt.grid()
plt.plot(train_scores.mean(axis=1), marker='o', c='blue')
plt.plot(valid_scores.mean(axis=1), marker='o', c='red')
plt.title('Train and evaluation scores')
plt.ylabel('ROC-AUC Score')
plt.xlabel('Progress')
plt.xticks([i for i in range(5)], np.linspace(0.1, 1, 5))
plt.legend(['Training score', 'Evaluation score'])
plt.show()

