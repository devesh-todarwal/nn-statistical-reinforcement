import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

seq_len = 128
#import X_train/X_val/X_test/y_train/y_val/y_test/train_data/test_data dfs

def create_model():
  in_seq = Input(shape=(seq_len, 5))

  x = Inception_A(in_seq, 32)
  x = Inception_A(x, 32)
  x = Inception_B(x, 32)
  x = Inception_B(x, 32)
  x = Inception_C(x, 32)
  x = Inception_C(x, 32)    
          
  x = Bidirectional(LSTM(128, return_sequences=True))(x)
  x = Bidirectional(LSTM(128, return_sequences=True))(x)
  x = Bidirectional(LSTM(64, return_sequences=True))(x) 
          
  avg_pool = GlobalAveragePooling1D()(x)
  max_pool = GlobalMaxPooling1D()(x)
  conc = concatenate([avg_pool, max_pool])
  conc = Dense(64, activation="relu")(conc)
  out = Dense(1, activation="sigmoid")(conc)      

  model = Model(inputs=in_seq, outputs=out)
  model.compile(loss="mse", optimizer="adam", metrics=['mae', 'mape'])     
  return model

model = create_model()

model.summary()

#uncomment to train

#callback = tf.keras.callbacks.ModelCheckpoint('MA-CNN+Bi-LSTM.hdf5', monitor='val_loss', save_best_only=True, verbose=1)

# model.fit(X_train, y_train,
#               batch_size=2048,
#               verbose=2,
#               callbacks=[callback],
#               epochs=200,
#               #shuffle=True,
#               validation_data=(X_val, y_val),)    

