import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv', header = None)

dataset = dataset.dropna(how = 'any', axis = 0)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

for i in range(0, 579):
    if y[i] == 2:
        y[i] = 0
        
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train[:, 1] = le.fit_transform(X_train[:, 1])
X_test[:, 1] = le.transform(X_test[:, 1])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

from keras.models import Sequential
from keras.layers import Dense, Dropout


classifier = Sequential()

classifier.add(Dense(output_dim = 128, activation = 'relu', input_dim = 10))
classifier.add(Dropout(0.4))
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 25, epochs = 100)

