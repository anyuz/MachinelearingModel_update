import pandas as pd
import xgboost as xgb
import gc
import numpy as np
import pickle as pickle
prop = pd.read_csv('properties_2016.csv')
train = pd.read_csv('train_2016_v2.csv')
sample = pd.read_csv('sample_submission.csv')

#get training and testing data from mySQL
#or get from website
# pull train and test data from S3 bucket

for c, dtype in zip(prop.columns, prop.dtypes):
	if dtype == np.float64:
		prop[c] = prop[c].astype(np.float32)


df_train = train.merge(prop, how='left', on='parcelid')

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 80000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

del x_train, x_valid; gc.collect()


params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

del d_train, d_valid


sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')

del prop; gc.collect()

x_test = df_test[train_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

del df_test, sample; gc.collect()

d_test = xgb.DMatrix(x_test)

del x_test; gc.collect()

print('Predicting on test ...')

p_test = clf.predict(d_test)

del d_test; gc.collect()
# read sample_submission.csv in S3
sub = pd.read_csv('sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test


pickle.dump(clf, open("xgboostmodel100.dat", "wb"))
# save model to S3 

# load model from file
#loaded_model = pickle.load(open("xgboostmodel.dat", "rb"))

# save predict result and score of model to S3
#sub.to_csv('predict.csv', index=False, float_format='%.4f') # Thanks to @inversion



