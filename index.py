#import
from cProfile import label
import pandas as pd;
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#data
data = pd.read_csv('InternetFirewallData.csv');
# print(data);
X = data.drop("Action", axis = 1);
y = data.Action;
# print(X);
# print(len(y.value_counts()));


#hold-out
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3.0, random_state = 100);
DT_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth = 10, min_samples_leaf = 15);
DT_gini.fit(X_train, y_train);

y_pred = DT_gini.predict(X_test);

print ("Accuracy is ", accuracy_score(y_test, y_pred) * 100);
mm = np.unique(y_test);
print ("array(",confusion_matrix(y_test, y_pred, labels = mm),")");

#Nghi thuc k-fold
KNN = KNeighborsClassifier(n_neighbors=10);
Bayes = GaussianNB();
kf = KFold(n_splits=100, shuffle = True, random_state = 6000);
acc_DT = 0;
acc_KNN = 0;
acc_Bayes = 0;
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index];
    y_train, y_test = y.iloc[train_index], y.iloc[test_index];
    #DT
    DT_gini.fit(X_train, y_train);
    y_pred_DT = DT_gini.predict(X_test);
    acc_DT += accuracy_score(y_test, y_pred_DT);
    #KNN
    KNN.fit(X_train, y_train);
    y_pred_KNN = KNN.predict(X_test);
    acc_KNN += accuracy_score(y_test, y_pred_KNN);
    #Bayes
    Bayes.fit(X_train, y_train);
    y_pred_Bayes = Bayes.predict(X_test);
    acc_Bayes += accuracy_score(y_test, y_pred_Bayes);
print("Độ chính xác trung bình DT: ", acc_DT/100*100);
print("Độ chính xác trung bình KNN: ", acc_KNN/100*100);
print("Độ chính xác trung bình Bayes: ", acc_Bayes/100*100);




