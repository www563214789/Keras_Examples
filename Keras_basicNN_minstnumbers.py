from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

#Step 1 Select Model type
model=Sequential()

#Step 2 Construct Neuron Network Layers
#2.1 Add input layer
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu')) 

#2.2 Add output layer
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

#Step 3 Compile
#sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #Set Optimizer arguments, Lr(Learning Rate)...etc
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

#Deal with input data
(x_train, y_train), (x_test, y_test) = mnist.load_data() #Use mnist tools from Keras to get datasets
#The dimension of input dataset in minst is (num,28,28), We need to transfer dimension 28,28 -> 784(28*28) (3D->2D)
x_train_2D = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]) #reshape( 資料筆數 , 目標維度數)
x_train_normal = x_train_2D.astype('float32')/255
x_test_2D = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_test_normal = x_test_2D.astype('float32')/255

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

#Step 4 Trainning
    #.fit 參數
    #batch_size：對總樣本數進行分組，每組包含的樣本數量
    #epochs ：訓練次數
    #shuffle：是否把訓練集隨機打亂後再進行訓練
    #validation_split：拿出百分之多少用來做交叉驗證
    #verbose： 0：不輸出  1：輸出進度  2：輸出每次的訓練結果

#Start Trainning
model.fit(x=x_train_normal, y=y_train_encoded, batch_size=800, epochs=10, verbose=2, validation_split=0.2)

#Evaluate for test datasets
scores = model.evaluate(x_test_normal, y_test_encoded)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))  

#Step 5 Output
X = x_test_normal[0:10,:]
predictions = model.predict_classes(X)
# get prediction result
print(predictions)
