import numpy as np
import pandas as pd


from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset=pd.read_csv('preprosseddata.csv')
X = dataset.iloc[:,[1,2,3,4,5,6,7,8]].values
Y = dataset.iloc[:,9].values
print(X[0])
dataset_scaled = sc.fit_transform(X)


dataset_scaled = pd.DataFrame(dataset_scaled)

X = dataset_scaled
Y = Y


'''dividing the dataset into train set and test set using train_test_split'''
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=42)

'''model'''
model= Sequential()
model.add(Dense(150,kernel_initializer='normal',activation='relu',input_dim=8))#input layer having 150 nodes
model.add(Dense(102,kernel_initializer='normal',activation='relu'))#hidden layer 1 having 102 nodes
model.add(Dense(50,kernel_initializer='normal',activation='relu'))#hidden layer 2 having 50 nodes
model.add(Dense(1,activation='sigmoid'))#out layer having one node
Adam=Adam(lr=0.0001)#learning rate==0.0001
model.compile(optimizer=Adam,loss='binary_crossentropy',metrics=['accuracy'])


#converting train,test datasets into numpy arrays before doing predition ,this is very important
xtrain = np.asarray(xtrain)
ytrain = np.asarray(ytrain)
xtest = np.asarray(xtest)
ytest = np.asarray(ytest)
#training
model.fit(xtrain,ytrain,batch_size=8,epochs=100,verbose=0)


trainscores=model.evaluate(xtrain,ytrain)#train score
testscores=model.evaluate(xtest,ytest)#test score
print('train accuracy=',trainscores)
print('test accuracy=',testscores)

'''saving model as .h5 format'''
model.save('model.h5')
print('model saved')

'''example'''
model1=load_model('C:\\Users\\Lenovo\\Desktop\\flaskapp\\model.h5')#loading the model
float_features = [1,85.0,66.0,29.0,155.5482233502538,26.6,0.35100000000000003,31]
final_features = [np.array(float_features)]
prediction = model1.predict(sc.transform(final_features))
pred=round(prediction[0][0])
if pred== 1:
    pred = "**You have Diabetes, please consult a Doctor."
elif pred== 0:
    pred = "**You don't have Diabetes."
output = pred
print(output)

