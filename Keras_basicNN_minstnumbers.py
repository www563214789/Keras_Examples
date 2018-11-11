from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

#Step 1 Select Model type
model=Sequential()

#Step 2 Construct Neuron Network Layers
#2.1 Input layer
model.add(Dense(units=500,input_dim=784)) #Input layer:28*28=784(data input from minst dataset)
model.add(Activation("relu")) #Activation function = tanh
model.add(Dropout(0.5)) #Set 50% dropout rate to prevent overfitting

#2.2 Dense layer
model.add(Dense(units=500)) #Dense layer node=500,same as input layer output nodes 
model.add(Activation("relu")) #Activation function = Tanh

#2.3 Output layer
model.add(Dense(units=10)) #Output Layer dimension is 10, means 10(represent number"0"-"9") classfication
model.add(Activation("softmax"))

#Step 3 Compile
#sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #Set Optimizer arguments, Lr(Learning Rate)...etc
#model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['accuracy']) #Use 交叉熵 as loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 


#Step 4 Trainning
    #.fit 參數
    #batch_size：對總樣本數進行分組，每組包含的樣本數量
    #epochs ：訓練次數
    #shuffle：是否把訓練集隨機打亂後再進行訓練
    #validation_split：拿出百分之多少用來做交叉驗證
    #verbose：屏显模式 0：不输出  1：输出进度  2：输出每次的训练结果

(x_train, y_train), (x_test, y_test) = mnist.load_data() #Use mnist tools from Keras to get datasets
#The dimension of input dataset in minst is (num,28,28), We need to transfer dimension 28,28 -> 784(28*28) (立體->平面)
x_train=x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_train = x_train.astype('float32')/255
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_test = x_test.astype('float32')/255

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#Start Trainning
model.fit(x=x_train, y=y_train, batch_size=200, epochs=50, verbose=0, validation_split=0.3)

#Evaluate for test datasets
model.evaluate(x_test, y_test, batch_size=200, verbose=0)

#Step 5 Output
print('Test set')
scores = model.evaluate(x_test, y_test, batch_size=200, verbose=0)
print(" ")
print("The test loss is %f" % scores)
result= model.predict(x_test,batch_size=200, verbose=0)

result_max = np.argmax(result, axis = 1)
test_max = np.argmax(y_test, axis = 1)

result_bool = np.equal(result_max, test_max)
true_num =  np.sum(result_bool)

print(" ")
print("The accuracy of the model is %f" % (true_num / len(result_bool)))
