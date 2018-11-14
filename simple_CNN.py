import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D #CNN 使用卷積層(Conv2D) 池化層(Pooling)
from keras.utils import np_utils

#Set image width and height
img_height = 28
img_width = 28

#Input data 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#input datatype of CNN model is different with simple neural network
#Channel mode for Tensorflow (amount, height, width, color channel(RGB=3 gray=1))
#Channel mode for Theano, Caffe (amount, color channel(RGB=3 gray=1), height, width)
x_train_2D = x_train.reshape(x_train.shape[0], img_width, img_height, 1)
x_test_2D = x_test.reshape(x_test.shape[0], img_width, img_height, 1)
input_shape = (img_width, img_height, 1)

x_train_2D = x_train_2D.astype('float32')/255
x_test_2D = x_test_2D.astype('float32')/255

y_train_encoded = keras.utils.to_categorical(y_train, 10)
y_test_encoded = keras.utils.to_categorical(y_test, 10)

#Construct a CNN model
model = Sequential()

#Input Layer
model.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu', input_shape=input_shape)) 

#First Convolution layer + Pooling layer
model.add(Conv2D(64, kernel_size=(3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Add Flatten layer to make dimension to 1
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

#Output layer
model.add(Dense(10, activation='softmax'))

#Compile model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

#Trainning
train_history = model.fit(x_train_2D, y_train_encoded, batch_size=128, epochs=12, verbose=2, validation_data=(x_test_2D, y_test_encoded))

#Evaluating
loss_rate, accu_rate = model.evaluate(x_test_2D, y_test_encoded, verbose=0)
print('Test loss:', loss_rate)
print('Test accuracy:', accu_rate)

#Saving model and weight
#Model
from keras.models import model_to_json
json_string = model_to_json #Get model architechture in json
with open("model.config","w") as text_file:
    text_file.write(json_string) #Create file and use function .write(filename) to save the model
#Weight
model.save_weights("model.weight") #function .save_weights(filename)

