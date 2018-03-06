import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
import time
from collections import Counter
import random
import sys

data_dir = '../data/'

df1 = pd.read_excel(data_dir + '2007_2011.xlsx', sheetname = 'Sheet1')
df2 = pd.read_excel(data_dir + '2012_2013.xlsx', sheetname = 'Sheet1')
df3 = pd.read_excel(data_dir + '2014.xlsx', sheetname = 'Sheet1')
df4 = pd.read_excel(data_dir + '2015.xlsx', sheetname = 'Sheet1')
df5 = pd.read_excel(data_dir + '2016_Q1.xlsx', sheetname = 'Sheet1')
df6 = pd.read_excel(data_dir + '2016_Q2.xlsx', sheetname = 'Sheet1')
df7 = pd.read_excel(data_dir + '2016_Q3.xlsx', sheetname = 'Sheet1')
df8 = pd.read_excel(data_dir + '2016_Q4.xlsx', sheetname = 'Sheet1')
df9 = pd.read_excel(data_dir + '2017_Q1.xlsx', sheetname = 'Sheet1')
df10 = pd.read_excel(data_dir + '2017_Q2.xlsx', sheetname = 'Sheet1')
df11 = pd.read_excel(data_dir + '2017_Q3.xlsx', sheetname = 'Sheet1')

date_since_2009 = datetime.date(2009, 1, 1)
df1 = df1[df1.issue_d > date_since_2009]
df_list = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11]
data = pd.concat(df_list, join = 'inner')
data = data[~data.loan_status.str.contains('Does not meet')]
data = data[data.loan_status != 'Issued']


train_data = pd.read_csv(data_dir + 'training_data.csv')
test_data = pd.read_csv(data_dir + 'testing_data.csv')
train_idx = train_data.id.tolist()
test_idx = test_data.id.tolist()

train_data_final = data[data.id.isin(train_idx)]
test_data_final = data[data.id.isin(test_idx)]
test_data_final.to_csv(data_dir + 'test_data.csv', index = False)
train_data_final.to_csv(data_dir + 'train_data.csv', index = False)

# Diff columns: ['term_num', 'label', 'cr_hist', 'duration', 'early_paid', 'ret']

train_data = pd.read_csv(data_dir + 'train.csv', parse_dates = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d', 'next_pymnt_d'])
test_data = pd.read_csv(data_dir + 'test.csv', parse_dates = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d', 'next_pymnt_d'])

# Exclude some columns that are not known beforehand
ex_col = ['last_pymnt_amnt', 'last_pymnt_d', 'mths_since_last_delinq', 'next_pymnt_d', 'pymnt_plan', 
          'total_pymnt', 'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp',
          'collection_recovery_fee', 'out_prncp', 'out_prncp_inv', 'recoveries']
# Drop these columns because they contains more than 80% missing values
ex_col += ['mths_since_last_record', 'mths_since_last_major_derog']
# Drop these columns because the their values are messy
ex_col += ['title', 'emp_title']
# Drop these columns because they contains only one value or useless information 
ex_col += ['application_type', 'last_credit_pull_d', 'desc']
# Use this column to calculate duration of loans
ex_col.remove('last_pymnt_d')
# Use this column to calculate total payment received
# Note that column funded amnt and tota pymnt both have no null values.
# We could also choose funded amnt inv instead of funded amnt, but 90% of values between these two have no difference.
ex_col.remove('total_pymnt')

def preprocess(data, ex_col):
    data = data.drop(ex_col, axis = 1)
    data.loc[data['home_ownership'] == 'ANY', 'home_ownership'] = "OTHER"
    data['label'] = ~data.loan_status.str.contains('Paid')
    data.label = data.label.astype(float)
    data['issue_year'] = data['issue_d'].map(lambda x: str(x.year))
    data['issue_month'] = data['issue_d'].map(lambda x: str(x.month))
    data['early_year'] = data['earliest_cr_line'].map(lambda x: str(x.year))
    data['early_month'] = data['earliest_cr_line'].map(lambda x: str(x.month))
    data['duration'] = (data.last_pymnt_d.values.astype('datetime64[M]') - data.issue_d.values.astype('datetime64[M]')) / np.timedelta64(1, 'M')
    data['ret'] = (data.total_pymnt - data.funded_amnt) / data.funded_amnt
    data = data.drop(['issue_d', 'earliest_cr_line', 'loan_status', 'last_pymnt_d', 'total_pymnt'], axis = 1)
    numeric_cols = [c for c in data.columns if data[c].dtype == 'float']
    other_cols = [c for c in data.columns if c not in numeric_cols]
    for c in numeric_cols:
        data[c] = data[c].fillna(0)
    return data, numeric_cols, other_cols

def tokenize_train(data, other_cols, min_count = 5):
    feature_dict = {}
    other_cols.remove('id')
    for col in other_cols:
        c = Counter(data[col])
        l = sorted(c.items(), key = lambda x: x[1], reverse = True)
        feature_dict[col] = {}
        for item in l:
            token = item[0] if item[1] >= min_count else 'UNK'
            if token not in feature_dict[col]:
                feature_dict[col][token] = len(feature_dict[col])
            else:
                continue
        # Assign id 0 to UNK tokens
        if 'UNK' in feature_dict[col]:
            data.loc[~data[col].isin(feature_dict[col]), col] = feature_dict[col]['UNK']
        data = data.replace({col: feature_dict[col]})
    data_dict = data.set_index('id').T.to_dict('dict')
    return data_dict, feature_dict

def tokenize_test(data, other_cols, feature_dict):
    for col in other_cols:
        if 'UNK' in feature_dict[col]:
            data.loc[~data[col].isin(feature_dict[col]), col] = feature_dict[col]['UNK']
        data = data.replace({col: feature_dict[col]})
    data_dict = data.set_index('id').T.to_dict('dict')
    return data_dict

clean_train, numeric_cols, other_cols = preprocess(train_data, ex_col)
clean_test, _, _ = preprocess(test_data, ex_col)
train_dict, feature_dict = tokenize_train(clean_train, other_cols)
test_dict = tokenize_test(clean_test, other_cols, feature_dict)

# Build validation set
valid_dict = {}
valid_keys = [k for k in train_dict if random.random() > 0.95]
for k in valid_keys:
    valid_dict[k] = train_dict[k]
    del train_dict[k] 


np.savez(data_dir + "final_data", train_dict = train_dict, valid_dict = valid_dict, test_dict = test_dict, feature_dict = feature_dict)

'''
How to decide min_count

for c in other_cols:
	a = Counter(clean_test[c])
	b = Counter(clean_train[c])
	if set(a.items()) == set(b.items()):
		continue
	print(c)
	try: 
		print(max([a[t] for t in a if t not in b]))
	except:
		pass
	try:
		print(max([b[t] for t in b if t not in a]))
	except:
		pass
'''
