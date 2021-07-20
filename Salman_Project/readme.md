```
# SVM
import pandas as pd
import numpy as np
from sklearn import svm


def clean_data(data):
    data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
    data['Age'] = data['Age'].fillna(data['Age'].dropna().median())

    data.loc[data['Sex'] == 'male', 'Sex'] = 0
    data.loc[data['Sex'] == 'female', 'Sex'] = 1

    data['Embarked'] = data['Embarked'].fillna('S')
    data.loc[data['Embarked'] == 'S', 'Embarked'] = 0
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'Q', 'Embarked'] = 2


train = pd.read_csv('train.csv')
target = train.Survived
clean_data(train)
features_name = ['Pclass', 'Fare', 'Embarked', 'Age', 'Sex', 'SibSp', 'Parch']
features = train[features_name]
# APPlying SVM model
model = svm.SVC(kernel='rbf', C=1, gamma=0.1)

model.fit(trainData, YTrain)
# Predictions
prediction2 = model.predict(testData)
# crossvalidation
Acc_n_Std = cross_val_score(model, trainData, YTrain, cv=10, n_jobs=-1)
print(Acc_n_Std)
print('Accuracy: ', np.mean(Acc_n_Std), ' Standard Deviation: ', np.std(Acc_n_Std))

submission2 = pd.DataFrame({
    "PassengerId": testData["PassengerId"],
    "Survived": prediction2
})
submission2.to_csv(r'C:\Users\Muhammad Salman Khan\Desktop\Ai_Assignment3\SVM.csv')
```
