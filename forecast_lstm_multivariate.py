from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from datetime import datetime
from sklearn.metrics import mean_squared_error
from scipy.ndimage.interpolation import shift
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from math import sqrt, floor
from matplotlib import pyplot as plt
from argparse import ArgumentParser 
from dataset import *
import glob


if __name__ == "__main__":
    # arguments
    parser = ArgumentParser(description="LSTM Multivariate Forecasting")
    parser.add_argument('-v', '--verbose', help='verbose mode', action="store_true")
    parser.add_argument('-g', '--train-on-validation', \
            help='train on validation set as well as training set', action="store_true")
    parser.add_argument('-r', '--split-rate', \
            help='training set split rate, reset would be validation', type=float, default=0.8)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch-size', type=int, default=1000)
    parser.add_argument('-u', '--lstm-units', type=int, default=100)
    parser.add_argument('-l', '--lstm-layers', type=int, default=1)
    parser.add_argument('-s', '--store_to_filter', \
            help='filter the store and plot prediction result', \
            nargs=1, default='air_754ae581ad80cc9f')
    parser.add_argument('--predict-test-set', \
            help='make prediction on test set and store result', action="store_true")
    parser.add_argument('--predict-train-set', \
            help='make prediction on train set', action="store_true")
    parser.add_argument('-n', '--name', \
            help='a name for the job, if model existing will continue fitting', required=True)
    args = parser.parse_args()

    # assert the input aarguments
    assert args.split_rate > 0.0 and args.split_rate <= 1.0 
    assert args.epochs > 0
    assert args.lstm_units > 1
    assert args.lstm_layers > 0 

    # execution mode report
    if args.verbose:
        print("--- Execution mode report ----")
        print("Fit model on All training set") if args.split_rate < 1.0 else print("%.2f training set size" % args.split_rate)
        print("running prediction for training set") if args.predict_train_set else None
        print("running prediction for validation set") if args.split_rate < 0.1 else None
        print("running prediction for test set") if args.predict_train_set else None
        print("plots for filtered store: ", args.store_to_filter) 

    # load ready to work with data
    train, test = load_data()

    # -------------- Choose a store for visualization purpose ----------------#
    # print(train['air_store_id'].tail(3100))
    store_to_filter = args.store_to_filter #"air_90f0efbb702d77b7" #"air_e9ebf7fc520ac76a"
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


    # drop unnecessary features
    x_train = train.drop(['air_store_id','visit_date','visitors'], axis=1)
    y_train = np.log1p(train['visitors'].values)
    if args.verbose:
        print("max visits: ", np.max(train['visitors'].values))
        print("min visits: ", np.min(train['visitors'].values))

    y_test = test['visitors'].as_matrix()
    x_test = test.drop(['id','air_store_id','visit_date','visitors'], axis=1)

    # Scale the network's input
    scaler = StandardScaler().fit(x_train) # MinMaxScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
        
    # Set random seed for reproduceability
    np.random.seed(7)

    if args.verbose:
        print("\n--- shape report ---")
        print("x_train: ", x_train.shape)
        print("y_train: ", y_train.shape)
        print("x_test: ", x_test.shape)

    # split the training data to train and validation
    train_sample_size = floor(x_train.shape[0]*args.split_rate)
    x_valid = np.copy(x_train[train_sample_size:,:])
    y_valid = np.copy(y_train[train_sample_size:])
    x_train = x_train[:train_sample_size,:]
    y_train = y_train[:train_sample_size]

    if args.split_rate == 1.0:
        assert x_valid.shape == (0, 50)
        assert y_valid.shape == (0,)

    # reshape for the network's input
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_valid = x_valid.reshape((x_valid.shape[0], 1, x_valid.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    if args.verbose:
        print("\n-- network input --")
        print("X_train: ", x_train.shape)
        print("y_train: ", y_train.shape)
        print("X_valid: ", x_valid.shape)
        print("y_valid: ", y_valid.shape)
        print("X_test: ", x_test.shape)

    model_files = glob.glob("%s-*.hdf5" % args.name)
    initial_epoch = 0
    model_file = ""
    model = None

    if len(model_files) > 0: # existing model 
        epoch_files = []
        for mf in model_files:
            s = mf.split("-")
            epoch = int(s[1])
            epoch_files.append((epoch, mf))
        epoch_files.sort(key = lambda t: t[0], reverse=True)
        initial_epoch, model_file = epoch_files[0]
        model = load_model(model_file)
    else: # new model
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
        if args.lstm_layers > 1:
            for i in range(args.lstm_layers-1):
                model.add(LSTM(args.lstm_units, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True ))
        model.add(LSTM(args.lstm_units, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dense(1))

    # print network info
    if args.verbose:
        print("\n--- Network Model Info ---")
        print("\tEpochs: %d" % args.epochs)
        print("\tBatch size: %d" % args.batch_size)
        print("\tLSTM units: %d" % args.lstm_units)
        print("\tLSTM layers: %d" % args.lstm_layers)

    adam = Adam(lr=0.001)
    model.compile(loss='mse', optimizer=adam)

    # create checkpoint for the model
    filepath = "%s-{epoch:02d}-{val_loss:.2f}.hdf5" % args.name
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, \
            save_best_only=True, mode='auto')#, period=1)
    callbacks = [checkpoint] # keras callbacks list 

    print("initial epoch: ", initial_epoch)

    # fit network
    if initial_epoch < args.epochs:
        v = 2 if args.verbose else 1
        if args.split_rate == 1.0: # train on all data
            print("fit the model on the whole training set") if args.verbose else None
            history = model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, \
                    verbose=v, shuffle=False, callbacks=callbacks, initial_epoch=initial_epoch)
        else: # train with validation 
            print("fit the model with validation") if args.verbose else None
            history = model.fit(x_train, y_train, epochs=args.epochs, \
                    batch_size=args.batch_size, validation_data=(x_valid, y_valid),\
                    verbose=v, shuffle=False, callbacks=callbacks, initial_epoch=initial_epoch)

        if args.split_rate < 1.0 and args.train_on_validation:
            history = model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, \
                    validation_data=(x_valid, y_valid), verbose=2, shuffle=False)
        # plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
    else:
        print("model is already trained") if args.verbose else None
    
    yhat = model.predict(x_valid)
    # yhat = model.predict(x_train)
    visitors = np.absolute(np.expm1(yhat))

    # make a prediction for test data
    if args.predict_test_set:
        yhat = model.predict(x_test)
        test['visitors'] = visitors 
        test[['id','visitors']].to_csv('.csv', index=False, float_format='%.3f')

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

    # --------------- cotninue of the plot from above -------------------- #
    fig = plt.figure()
    fig.suptitle('input features for store: %s' % store_to_filter)

    for i, col in enumerate(col_to_plot):
        # ttt = pd.Series(data[:,i], index=data[:,1]) # TODO
        ax = plt.subplot(len(col_to_plot), 1, i+1)
        if col == 'visitors':
            t1 = list(range(0, data.shape[0]))
            this_store_2 = this_store[-len(visitors):]
            pred_for_this_store = visitors[this_store_2]
            # print(pred_for_this_store.shape)
            t2 = list(range(data.shape[0]-len(pred_for_this_store), data.shape[0]))
            # print(t1)
            # print(t2)
            # plt.plot(t1, data[:, i], t2, pred_for_this_store)
            ax.plot(t1, data[:,i], label='original')
            ax.plot(t2, pred_for_this_store, label='prediction')
        else:
            plt.plot(data[:, i])
        # ttt.plot()
        plt.legend(loc='upper left')
        plt.title(col, y=0.5, loc='right')

    plt.xlabel('Time step')
    plt.show()
    # --------------- end of the plot from above -------------------- #

    # plot on validation set 
    if args.split_rate < 1.0: # if there was a validation splitage
        # plot on validation and prediction
        plt.plot(np.absolute(np.expm1(y_valid)))
        # plt.plot(np.absolute(np.expm1(y_train)))
        plt.plot(visitors)
        plt.show()
