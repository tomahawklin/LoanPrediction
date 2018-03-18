import numpy as np
import random
from utils import load_data, batch_iter, set_cuda, detach_cuda, get_stats, con_regs_strtgy
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, accuracy_score
import time
import math

train_data, valid_data, test_data, embed_dict, embed_dims, embed_keys, float_keys = load_data('../data/final_data.npz')

topk = 1000

# Formalize the data
train_ids = sorted([k for k in train_data])
X_train = np.array([[train_data[d][k] for k in float_keys] + [train_data[d][k] for k in embed_keys] for d in train_ids])
label_train = np.array([train_data[k]['label'] for k in train_ids])
duration_train = np.array([train_data[k]['duration'] for k in train_ids])
ret_train = np.array([train_data[k]['ret'] for k in train_ids])
valid_ids = sorted([k for k in valid_data])
X_valid = np.array([[valid_data[d][k] for k in float_keys] + [valid_data[d][k] for k in embed_keys] for d in valid_ids])
label_valid = np.array([valid_data[k]['label'] for k in valid_ids])
ret_valid = np.array([valid_data[k]['ret'] for k in valid_ids])
test_ids = sorted([k for k in test_data])
X_test = np.array([[test_data[d][k] for k in float_keys] + [test_data[d][k] for k in embed_keys] for d in test_ids])
label_test = np.array([test_data[k]['label'] for k in test_ids])
ret_test = np.array([test_data[k]['ret'] for k in test_ids])

# Randomforest classifer as a benchmark for classification method

clf = RandomForestClassifier(max_depth = 10, random_state = 0)
clf.fit(X_train, label_train)
label_pred = clf.predict(X_test)
pos_prob = clf.predict_proba(X_test)[:, 1]
print(roc_auc_score(label_test, pos_prob), accuracy_score(label_test, label_pred))
# 0.956398013417 0.915384649479
indices = (1 - pos_prob).argsort()[-topk:][::-1]
rets = np.take(ret_test, indices)
print(np.mean(rets), np.median(rets), np.std(rets))
# 0.0629157587959 0.0654337746333 0.0575278553421

# Linear regression with regularization as a benchmark for regression method
clf = linear_model.Lasso(alpha = 0.001)
clf.fit(X_train, ret_train)
ret_pred = clf.predict(X_test)
indices = ret_pred.argsort()[-topk:][::-1]
rets = np.take(ret_test, indices)
print(np.mean(rets), np.median(rets), np.std(rets))
# 0.171195174684 0.172147034292 0.103159394291
# Stronger L1 regularization makes performance worse

clf = linear_model.Ridge(alpha = 0.01)
clf.fit(X_train, ret_train)
ret_pred = clf.predict(X_test)
indices = ret_pred.argsort()[-topk:][::-1]
rets = np.take(ret_test, indices)
print(np.mean(rets), np.median(rets), np.std(rets))
# 0.195486604439 0.197774390532 0.106759470241
# Performance does not seem to be sensitive to L2 regularization 

from sklearn.svm import SVR
clf = SVR()
clf.fit(X_train, ret_train)
ret_pred = clf.predict(X_test)
indices = ret_pred.argsort()[-topk:][::-1]
rets = np.take(ret_test, indices)
print(np.mean(rets), np.median(rets), np.std(rets))

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(max_depth = 10, random_state = 0)
clf.fit(X_train, ret_train)
ret_pred = clf.predict(X_test)
indices = ret_pred.argsort()[-topk:][::-1]
rets = np.take(ret_test, indices)
print(np.mean(rets), np.median(rets), np.std(rets))
# 0.398549420488 0.45823683118 0.224799514101

from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor(max_depth = 10, random_state = 0)
clf.fit(X_train, ret_train)
ret_pred = clf.predict(X_test)
indices = ret_pred.argsort()[-topk:][::-1]
rets = np.take(ret_test, indices)
print(np.mean(rets), np.median(rets), np.std(rets))
# 0.378766483272 0.447832477229 0.262912654323

# Random forest is more robust compared to decision tree regressor