# 0 引用函式庫
from keras.models import Sequential #循序模型 
from keras.layers import Dense,Activation,Dropout #隱藏層(Dense) 激勵函數(Activation function) 隨機斷開(Dropout)
from keras.optimizers import SGD #優化器 梯度下降(SGD), Adam 等
from keras.datasets import mnist #載入 Minst 資料集
from keras.utils import to_categorical #資料標籤編碼
import numpy as np  #矩陣運算

# 1 建立模型
model = Sequential()

# 2 建構神經網路層
    # .add(Dense())
    # units : 輸出變數(神經單元)
    # input_dim : 輸入維度  
    # 維度輸入格式 
    # input_shape=(784,) = input_dim=784 = batch_input_shape=(None, 784)))
    # activation : 激勵函數 relu, tanh, softmax ...等

# 2.1 加入 輸入層 (Input layer) 
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu')) 

# 2.2 加入 隱藏層 (Dense layer) //基礎神經網路模型省略

# 2.3 加入 輸出層 (Output layer)
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

# 3 編譯網路模型 (載入已存檔模型與權重一樣需要編譯)
    # loss : 損失函數
    # optimizer : 優化器 (選擇計算損失函數的方法)
    # metrics : 衡量網路好壞的標準 accuracy, precision, recall, ...等
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

# 處理輸入資料
(x_train, y_train), (x_test, y_test) = mnist.load_data() #已從mnist載入資料集為例 也可以自行準備
# mnist載入的資料集
    # x_train : 訓練圖片
    # y_train : 訓練圖片標籤
    # x_test : 測試圖片
    # y_test : 側視圖片標籤
# [資料] 因為從mnist載入的資料集型態為3D(資料筆數, 高, 寬) 但需要把資料轉成2D型態(資料筆數, 高*寬) 才能送入輸入層
x_train_2D = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]) # reshape( 資料筆數 , 目標維度數)
x_train_normal = x_train_2D.astype('float32') / 255 #將 0 - 255 轉成 0 - 1
x_test_2D = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_test_normal = x_test_2D.astype('float32') / 255

# [標籤] 資料集標籤編碼
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# 4 訓練神經網路
    #.fit() 參數
    # batch_size：對總樣本數進行分組，每組包含的樣本數量
    # epochs：訓練次數
    # shuffle：是否把訓練集隨機打亂後再進行訓練 True/False
    # validation_split：拿出百分之多少用來做交叉驗證
    # verbose：0：不輸出  1：輸出進度  2：輸出每次的訓練結果

model.fit(x=x_train_normal, y=y_train_encoded, batch_size=800, epochs=10, verbose=2, validation_split=0.2)

# 使用 測試集 驗證訓練過後的網路模型 
    # evaulate(測試集, 測試集編碼)
    # return (損失百分比, 準確率)
loss_rate, accu_rate = model.evaluate(x_test_normal, y_test_encoded)  
print()  
print("\t[Info] Loss rate of testing data = {:2.1f}%".format(loss_rate *100.0))
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(accu_rate *100.0))  


# 5 測試集丟入神經網路測試預測結果
    # predict_classes : 回傳分類結果 [分類]
    # predict : 回傳分類結果的數值 [數值] -> 需調用 argmax() 做後續處理
X = x_test_normal[0:20,:]
predictions = model.predict_classes(X)
# 顯示預測結果
print(predictions)

# 6 神經網路存檔
# 模型結構(Network Architechture)存檔
from keras.models import model_from_json
json_string = model.to_json()
with open("model.config", "w") as text_file:
    text_file.write(json_string)

# 模型訓練權重結果存檔
model.save_weights("model.weight")

# [Testing] 結構與權重同時存檔 格式 HDF5
#from keras import load_model
#model.save('model.h5')

