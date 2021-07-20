![LR](https://user-images.githubusercontent.com/86187568/126339432-119160ca-b139-4414-9f1a-d7994c7ab8b6.jpg)

#linear classification
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import LogisticRegression #library for model

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
trainData.drop('Sex',inplace=True,axis=1)
testData.drop('Name',inplace=True,axis=1)
testData.drop('Cabin',inplace=True,axis=1)
testData.drop('Ticket',inplace=True,axis=1)
testData.drop('Sex',inplace=True,axis=1)
#replace nan with mean value 
trainData['Fare']=trainData['Fare'].fillna(trainData['Fare'].dropna().mean())
trainData['Age']=trainData['Age'].fillna(trainData['Age'].dropna().mean())
#Convert String based columns to integer classes
trainData["Embarked"] = trainData["Embarked"].replace(['S','Q','C'],[10,20,30])
testData["Embarked"] = testData["Embarked"].replace(['S','Q','C'],[10,20,30])


#Remove Nan Values from Train
trainData.fillna(value=trainData['Age'].mean(),inplace=True)
trainData.fillna(value=trainData['Fare'].mean(),inplace=True)
trainData['Embarked']=trainData['Embarked'].fillna(20)
#Dummy values in Test for all NaN
testData.fillna(value=trainData['Age'].mean(),inplace=True)
testData.fillna(value=trainData['Fare'].mean(),inplace=True)

#Test
print(YTrain.shape)
print(trainData)
print(testData.shape)
#appying linear model
logisticRegr = LogisticRegression()
logisticRegr.fit(trainData,YTrain)
#Predictions
prediction3 = logisticRegr.predict(testData)
#crossvalidation
Acc_n_Std = cross_val_score(logisticRegr, trainData,YTrain, cv=10, n_jobs = -1)
print(Acc_n_Std)
print('Accuracy: ', np.mean(Acc_n_Std), ' Standard Deviation: ', np.std(Acc_n_Std))

submission3 = pd.DataFrame({
        "PassengerId": testData["PassengerId"],
        "Survived": prediction3
    })
submission3.to_csv(r'C:\Users\Rabiya Rahman\Desktop\Dataset\LR.csv')
