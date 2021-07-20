![AI_Final_Project](https://user-images.githubusercontent.com/82214700/126353201-5fd4966d-8c0c-4ea6-a255-04039c4d90fd.PNG)
#knn
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

#Read CSV Files
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

#Get Labels and remove them from Train data
YTrain = trainData.Survived;
trainData.drop('Survived',inplace=True,axis=1)

#Convert Data to Features
trainData.drop('Name',inplace=True,axis=1)
trainData.drop('Cabin',inplace=True,axis=1)
trainData.drop('Ticket',inplace=True,axis=1)
testData.drop('Name',inplace=True,axis=1)
testData.drop('Cabin',inplace=True,axis=1)
testData.drop('Ticket',inplace=True,axis=1)
#Convert String based columns to integer classes
trainData["Sex"] = trainData["Sex"].replace(['female','male'],[0,1])
testData["Sex"] = testData["Sex"].replace(['female','male'],[0,1])
trainData["Embarked"] = trainData["Embarked"].replace(['S','Q','C'],[0,1,2])
testData["Embarked"] = testData["Embarked"].replace(['S','Q','C'],[0,1,2])

#Remove Nan Values from Train
trainData.fillna(value=0,inplace=True)
#Dummy values in Test for all NaN
testData.fillna(value=trainData['Age'].mean(),inplace=True)
testData.fillna(value=trainData['Fare'].mean(),inplace=True)

#Test
print(YTrain.shape)
print(trainData)
print(testData.shape)
#applying KNN Model
knn = KNeighborsClassifier()

knn.fit(trainData,YTrain)
#Predictions
predictions = knn.predict(testData)
#cross validation applied
Acc_n_Std = cross_val_score(knn, trainData,YTrain, cv=10, n_jobs = -1)
print(Acc_n_Std)
print('Accuracy: ', np.mean(Acc_n_Std), ' Standard Deviation: ', np.std(Acc_n_Std))

submission = pd.DataFrame({
        "PassengerId": testData["PassengerId"],
        "Survived": predictions
    })
submission.to_csv(r'C:\Users\COMPUTERS WAY\Desktop\Saad_Project\KNN.csv')
