from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512)
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(Y_train.shape[1]))
model.add(Activation('softmax'))

# コンパイル
model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

# callback
es = EarlyStopping(monitor = 'val_loss', min_delta = 0.0000, patience = 5)
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=256, epochs=100, callbacks=[es])

# データの可視化（テストデータの先頭十枚)
for i in range(10):
  plt.subplot(2, 5, i+1)
  plt.imshow(X_test[i])
plt.subtitle("10 image of test data", fonsize=20)
plt.show()

pred = np.argmax(model.predict(X_test[10], axis=1))
print(pred)

model.summary()