import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from sklearn import preprocessing
from contextlib import contextmanager
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import pandas as pd
import time
import numpy as np

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def build_model(train_dataset):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=[train_dataset.shape[1]]))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    optimizer = tf.keras.optimizers.Adam(0.005)
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae', 'mse'])
    print(model.summary())
    return model  

def tf_dense_model(x_train,y_train,x_val,y_val,features,x_test):
    X_train, X_valid, X_test = x_train[features].reset_index(drop=True).copy(),\
                               x_val[features].reset_index(drop=True).copy(),\
                               x_test[features].reset_index(drop=True).copy()
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train.values)
    X_valid = sc.transform(X_valid.values)
    X_test = sc.transform(X_test.values)
        
    sc = StandardScaler()
    y_train = sc.fit_transform(y_train.values.reshape(-1, 1))
    y_val =sc.transform(y_val.values.reshape(-1, 1))
    
    model = build_model(X_train)

    callbacks = [
        ReduceLROnPlateau(), 
        EarlyStopping(patience=10), 
        ModelCheckpoint(filepath='mode_best.h5', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs', profile_batch=0)
    ]

    model.fit(X_train, y_train,
          validation_data=(X_valid, y_val),
          epochs=50,
          batch_size=256,
          verbose=1,
          shuffle=True,
          callbacks=callbacks)
    
    pred_test = model.predict(X_test)[:, 0]
    pred_test = sc.inverse_transform(pred_test)
    pred_train = model.predict(X_train)[:, 0]
    pred_train = sc.inverse_transform(pred_train)
    pred_val = model.predict(X_valid)[:, 0]
    pred_val = sc.inverse_transform(pred_val)
    return pred_test,pred_train,pred_val