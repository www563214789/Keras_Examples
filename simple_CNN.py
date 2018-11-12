from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.core import Activation
from keras.optimizer import Adadelta()
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D #CNN 使用卷積層(Conv2D) 池化層(Pooling)

img_height = 28
img_width = 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_2D = x_train.reshape(x_train.shape[0], img_width, img_height, 1)
x_test_2D = x_test.reshape(x_test.shape[0], img_width, img_height, 1)
input_shape = (img_width, img_height, 1)

x_train_2D = x_test_2D.astype('float32')/255
x_test_2D = x_test_2D.astype('float32')/255

y_train_encoded = np_utils.to_categorical(y_train, 10)
y_test_encoded = np_utils.to_categorical(y_test, 10)

#建構模型
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation = 'relu')
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(loss='keras.losses.categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

train_history = model.fit(x_train_2D, y_train_encoded, batch_size=100, epochs=12, verbose=2, validation_data=(x_test_2D, y_test_encoded))

loss_rate, accu_rate = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', loss_rate)
print('Test accuracy:', accu_rate)






