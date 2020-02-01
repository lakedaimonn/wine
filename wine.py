from keras.models import Sequential
from keras.layers import Dense

import pandas as pd

df = pd.read_csv('wine.csv', header=None)
df = df.sample(frac=1)
data_set = df.values

X = data_set[:, 0:12]
Y = data_set[:, 12]

model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(X, Y, epochs=200, batch_size=200)

print('\nAccuracy: {:.4f}'.format(model.evaluate(X, Y)[1]))

from keras.models import load_model

# save model
model.save('snowdeer_model.h5')

# load model
model = load_model('snowdeer_model.h5')

from keras.models import load_model

# save model
model.save('snowdeer_model.h5')

# load model
model = load_model('snowdeer_model.h5')

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import os
import pandas as pd

MODEL_SAVE_FOLDER_PATH = './model/'

df = pd.read_csv('wine.csv', header=None)
df = df.sample(frac=1)
data_set = df.values

X = data_set[:, 0:12]
Y = data_set[:, 12]

if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'

cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                verbose=1, save_best_only=True)

model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(X, Y, validation_split=0.2, epochs=200, batch_size=200, verbose=0,
          callbacks=[cb_checkpoint])

print('\nAccuracy: {:.4f}'.format(model.evaluate(X, Y)[1]))

from keras.callbacks import EarlyStopping

# ...

cb_early_stopping = EarlyStopping(monitor='val_loss', patience=100)

# ...

model.fit(X, Y, validation_split=0.2, epochs=5000, batch_size=500,
          callbacks=[EarlyStopping])
