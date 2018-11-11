from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np

#Step 1 Select Model type
model=Sequential()

#Step 2 Construct Neuron Network Layers
#2.1 Input layer
model.add(Dense(units=500,input_dim=784)) #Input layer:28*28=784(data input from minst dataset)
model.add(Activation("tanh")) #Activation function = tanh
model.add(Dropout(0.5)) #Set 50% dropout rate to prevent overfitting

#2.2 Dense layer
model.add(Dense(units=500)) #Dense layer node=500,same as input layer output nodes 
model.add(Activation("tanh")) #Activation function = Tanh

#2.3 Output layer
model.add(Dense(units=10)) #Output Layer dimension is 10, means 10(represent number"0"-"9") classfication
model.add(Activation("softmax"))

#Step 3 Compile
sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #Set Optimizer arguments, Lr(Learning Rate)...etc
model.compile(loss='categorical_crossentropy',optimizer='sgd',class_mode='categorical', metrics=['accuracy']) #Use 交叉熵 as loss function

#Step 4 Trainning
    #.fit的一些参数
    #batch_size：对总的样本数进行分组，每组包含的样本数量
    #epochs ：训练次数
    #shuffle：是否把数据随机打乱之后再进行训练
    #validation_split：拿出百分之多少用来做交叉验证
    #verbose：屏显模式 0：不输出  1：输出进度  2：输出每次的训练结果

(x_train, y_train), (x_test, y_test)=minst.load_data() #Use minst tools from Keras to get datasets
#The dimension of input dataset in minst is (num,28,28), We need to transfer dimension 28,28 -> 784(28*28) (立體->平面)
x_train=x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
y_train=(np.arange(10) == y_train[:,None]).astype(int)
y_test=(np.arange(10) == y_test[:,None]).astype(int)

model.fit(x_train, y_train, batch_size=200, epochs=50, shuffle=True, verbose=0, validation_split=0.3)
mode.evaluate(x_test, y_test,batch_size=200, verbose=0)

#Step 5 Output
print('Test set')
scores=model.evaluate(x_test, y_test, batch_size=200, verbose=0)
print(" ")
print("The test loss is %f" % scores)
result= model.predict(x_test,batch_size=200, verbose=0)

result_max = np.argmax(result, axis = 1)
test_max = np.argmax(y_test, axis = 1)

result_bool = np.equal(result_max, test_max)
true_num =  np.sum(result_bool)

print(" ")
print("The accuracy of the model is %f" % (true_num / len(result_bool)))


model.fit(x_train,y_train,epochs=5,batch_size=32)