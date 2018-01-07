import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.ndimage.interpolation import shift
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv2D, MaxPooling2D
from math import sqrt, floor
from matplotlib import pyplot as plt


data = {
    'tra': pd.read_csv('data/air_visit_data.csv'),
    'as': pd.read_csv('data/air_store_info.csv'),
    'hs': pd.read_csv('data/hpg_store_info.csv'),
    'ar': pd.read_csv('data/air_reserve.csv'),
    'hr': pd.read_csv('data/hpg_reserve.csv'),
    'id': pd.read_csv('data/store_id_relation.csv'),
    'tes': pd.read_csv('data/sample_submission.csv'),
    'hol': pd.read_csv('data/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])


print("max visits: ", np.max(data['hr']['reserve_visitors'].values))
print("min visits: ", np.min(data['hr']['reserve_visitors'].values))

print("max visits: ", np.max(data['tra']['visitors'].values))
print("min visits: ", np.min(data['tra']['visitors'].values))

for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})
    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)
#OPTIMIZED BY JEROME VALLET
tmp = data['tra'].groupby(['air_store_id','dow']).agg({'visitors' : [np.min,np.mean,np.median,np.max,np.size]}).reset_index()
tmp.columns = ['air_store_id', 'dow', 'min_visitors', 'mean_visitors', 'median_visitors','max_visitors','count_observations']
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 
# NEW FEATURES FROM Georgii Vyshnia
stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))
stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))
lbl = LabelEncoder()
for i in range(10):
    stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
    stores['air_area_name'+str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 

train = pd.merge(train, stores, how='left', on=['air_store_id','dow']) 
test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])

for df in ['ar','hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 
    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])

# train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2

# NEW FEATURES FROM JMBULL
train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
train['var_max_lat'] = train['latitude'].max() - train['latitude']
train['var_max_long'] = train['longitude'].max() - train['longitude']
test['var_max_lat'] = test['latitude'].max() - test['latitude']
test['var_max_long'] = test['longitude'].max() - test['longitude']

# NEW FEATURES FROM Georgii Vyshnia
train['lon_plus_lat'] = train['longitude'] + train['latitude'] 
test['lon_plus_lat'] = test['longitude'] + test['latitude']

lbl = LabelEncoder()
train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
test['air_store_id2'] = lbl.transform(test['air_store_id'])

col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date','visitors']]
train = train.fillna(-1)
test = test.fillna(-1)

print(train['air_store_id'].tail(3100))
# -----------------------------------------------------------------------------#
store_to_filter = "air_754ae581ad80cc9f"#"air_90f0efbb702d77b7"#"air_e9ebf7fc520ac76a"
this_store = train['air_store_id'] == store_to_filter
col_to_plot = [
        'visitors', 
        'day_of_week',
        'holiday_flg', 
        'mean_visitors', 
        'count_observations',
        # 'min_visitors', 
        # 'median_visitors', 
        # 'max_visitors', 
        # 'rs1_x', 'rv1_x', 'rs2_x', 'rv2_x', 
        # 'rs1_y', 'rv1_y', 'rs2_y', 'rv2_y', 
        # 'total_reserv_sum',
        'total_reserv_mean'
        # 'total_reserv_dt_diff_mean', 
         # 'date_int'
        ]

train_plt = train[col_to_plot]
data = train_plt.values[this_store, :]
# I moved the plot to the end of the operation so that I can put the prediction
# for that specific store as well

# -----------------------------------------------------------------------------#

x_train = train.drop(['air_store_id','visit_date','visitors'], axis=1)
y_train = np.log1p(train['visitors'].values)
print("max visits: ", np.max(train['visitors'].values))
print("min visits: ", np.min(train['visitors'].values))

# yhat = np.absolute(np.expm1(y_train))
# plt.plot(yhat, label='reversed')
# plt.plot(train['visitors'].values, label='original')
# plt.show()
# exit(0)

y_test = test['visitors'].as_matrix()
x_test = test.drop(['id','air_store_id','visit_date','visitors'], axis=1)

print(x_train.columns.values)
print(x_train.head())
print("-----")
# print(y_train.columns.values)
print(x_test.columns.values)
print(x_test.head())
# print(y_test.columns.values)

# Define the scaler 
scaler = StandardScaler().fit(x_train)
# scaler = MinMaxScaler().fit(x_train)

# Scale the train set
x_train = scaler.transform(x_train)

# Scale the test set
x_test = scaler.transform(x_test)
    
# Set random seed
np.random.seed(7)

# print("y_train before: ", y_train)
# y_train = shift(y_train, -1)
# print("y_train after: ",y_train)

print("--- shape report ---")
print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)

rate = 0.8
train_sample_size = floor(x_train.shape[0]*rate)
# to have a validation
# print(type(x_train))
x_valid = np.copy(x_train[train_sample_size:,:])
y_valid = np.copy(y_train[train_sample_size:])
x_train = x_train[:train_sample_size,:]
y_train = y_train[:train_sample_size]

x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_valid = x_valid.reshape((x_valid.shape[0], 1, x_valid.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

print("-- network input --")
print("X_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("X_valid: ", x_valid.shape)
print("y_valid: ", y_valid.shape)
print("X_test: ", x_test.shape)

# design network
model = Sequential()
# try1:
# model.add(Conv2D(input_shape=(x_train.shape[1], x_train.shape[2]), filters=32, kernel_size=3, padding='same', activation='tanh'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(LSTM(100))
# try2:
# model.add(LSTM(256, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True ))
# model.add(LSTM(256, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True ))
# model.add(LSTM(100))
# try3:
model.add(LSTM(100, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# fit network
history = model.fit(x_train, y_train, epochs=10, batch_size=1000, \
        validation_data=(x_valid, y_valid), verbose=2, shuffle=False)
# history = model.fit(x_train, y_train, epochs=100, batch_size=1000, \
                    # verbose=2, shuffle=False)

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
 
# make a prediction for test data
# yhat = model.predict(x_test)
yhat = model.predict(x_valid)
# yhat = model.predict(x_train)
visitors = np.absolute(np.expm1(yhat))
# test['visitors'] = visitors 
# test[['id','visitors']].to_csv('2lstm100_submission_1.csv', index=False, float_format='%.3f')

# some debuggings with validation data
# print("#"*20)
# print("yvalid: ", y_valid)
# print("yhat: ", yhat)
# print("visitors: ", visitors)
# print(type(visitors))
# print("visitor.shape:", visitors.shape)
# print(type(y_valid))
# print("y_valid.shape: ", y_valid.shape)
# print("#"*20)

# calculate RMSE for the validation set
rmse = sqrt(mean_squared_error(yhat, y_valid))
# rmse = sqrt(mean_squared_error(yhat, y_train))
print('Test RMSE: %.3f' % rmse)

# --------------- cotninue of the plot from top -------------------- #
fig = plt.figure()
fig.suptitle('input features for store: %s' % store_to_filter)

for i, col in enumerate(col_to_plot):
    # ttt = pd.Series(data[:,i], index=data[:,1]) # TODO
    plt.subplot(len(col_to_plot), 1, i+1)
    if col == 'visitors':
        t1 = list(range(0, data.shape[0]))
        this_store_2 = this_store[-len(visitors):]
        pred_for_this_store = visitors[this_store_2]
        print(pred_for_this_store.shape)
        t2 = list(range(data.shape[0]-len(pred_for_this_store), data.shape[0]))
        print(t1)
        print(t2)
        plt.plot(t1, data[:, i], t2, pred_for_this_store)
    else:
        plt.plot(data[:, i])
    # ttt.plot()
    plt.title(col, y=0.5, loc='right')
plt.xlabel('Time step')
plt.show()
# --------------- end of the plot from top -------------------- #


# plot on validation adn prediction
plt.plot(np.absolute(np.expm1(y_valid)))
# plt.plot(np.absolute(np.expm1(y_train)))
plt.plot(visitors)
plt.show()
