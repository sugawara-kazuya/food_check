from tensorflow improt keras
from tensorflow.keras import optimizers
from tensorflow.keras.datasets improt cifar10
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequentail, load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.application.vgg16 import vgg16
from tensorflow.keras import Input, layers, Model
from keras.callbacks import EarlyStopping

# トレーニングデータとtestデータを分ける
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# モデルの定義
input_tensor = Input(shape=(200, 200, 3))

vgg16 = VGG16(include_top=False, weifhts='imagenet', input_tensor=input_tensor)

top_model = Sequentail()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(101, activation='softmax'))

# vgg16とtop_modelを連結する
model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

# vgg16による特徴量部分の重みは更新されると崩れてしまうので固定
for layer in model.layers[:19]:
    layer.trainable = False

# コンパイル
model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

# calback関数の定義
es = EarlyStopping(monitor = 'val_loss', min_delta=0.0000, patience=5)

#トレーニングデータの学習
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=256, epochs=100, callbacks = [es])

# 精度評価
scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss', scores[0])
print('Test accuracy', scores[1])

# データの可視化（テストデータの先頭の10枚）
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i])
plt.subtitle("10 image of test data", fortsize=20)
plt.show()

#予測
pred = np.argmax(model.predict(X_test[0:10]), axis=1)
print(pred)

model.summary()