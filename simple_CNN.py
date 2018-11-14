import keras
from keras.datasets import mnist #Use mnist dataset
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
#CNN use convolution layer(Conv2D) and Pooling layer(MaxPooling2D)
from keras.layers.convolutional import Conv2D, MaxPooling2D 
from keras.utils import np_utils

#Set image width and height
img_height = 28
img_width = 28

#Input data 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#input datatype of CNN model is different with simple neural network
#[Channel] Mode for backend engine
#Channel mode for Tensorflow (amount, height, width, color channel(RGB=3 gray=1))
#Channel mode for Theano, Caffe (amount, color channel(RGB=3 gray=1), height, width)
x_train_2D = x_train.reshape(x_train.shape[0], img_height, img_width, 1)
x_test_2D = x_test.reshape(x_test.shape[0], img_height, img_width, 1)
input_shape = (img_height, img_width, 1)
#Normalize value in 0-1 in float32
x_train_2D = x_train_2D.astype('float32')/255
x_test_2D = x_test_2D.astype('float32')/255

#Encode labels to 0-9 in like 0000100000 means 4 (at 4th position of bit)
#In trainning data ====> [Image]->[Label]
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

#Evaluating accuracy rate and loss rate
loss_rate, accu_rate = model.evaluate(x_test_2D, y_test_encoded, verbose=0)
print('Test loss:', loss_rate)
print('Test accuracy:', accu_rate)

#Put test dataset in trained neural network and get predicted result
x = x_test_2D[:20, :]
prediction = model.predict_classes(X)
print(prediction)

#Saving model and weight
from keras.models import model_from_json
#Model
json_string = model.to_json() #Get model architechture in json by model_to_json()
with open("model.config","w") as text_file:
    text_file.write(json_string) #Create file and use function .write(filename) to save the model
#Weight
model.save_weights("model.weight") #function .save_weights(filename)

