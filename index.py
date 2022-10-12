#import
from cProfile import label
import pandas as pd;
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

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
kf = KFold(n_splits=100, shuffle = True, random_state = 6000);
acc_DT = 0;
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index];
    y_train, y_test = y.iloc[train_index], y.iloc[test_index];
    DT_gini.fit(X_train, y_train);
    y_pred = DT_gini.predict(X_test);
    acc_DT += accuracy_score(y_test, y_pred);
print("Độ chính xác trung bình: ", acc_DT/100*100);




