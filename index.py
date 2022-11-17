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
# print(data);
# print(len(y.value_counts()));


# hold-out
i = 0;
j = 0;
for i in range(91, 101, 1) :
    # for j in range(1, 11, 1) :
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3.0, random_state = i);
    DT_gini = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 8, min_samples_leaf = 3);
    DT_gini.fit(X_train, y_train);
    y_pred_DT = DT_gini.predict(X_test);
    # print("Độ chính xác [", i ,"][", j, "]: ", accuracy_score(y_test, y_pred_DT)*100);
    
    #Huan luyen mo hinh KNN
    KNN = KNeighborsClassifier(n_neighbors=10);
    KNN.fit(X_train, y_train);
    
    #Huan luyen mo hinh Bayes
    Bayes = GaussianNB();
    Bayes.fit(X_train, y_train);
    
    y_pred_DT = DT_gini.predict(X_test);
    y_pred_KNN = KNN.predict(X_test);
    y_pred_Bayes = Bayes.predict(X_test);
    # mm = np.unique(y_test);
    print("Độ chính xác trung bình DT", i, ": ", accuracy_score(y_test, y_pred_DT)*100);
    # print ("array(\n",confusion_matrix(y_test, y_pred_DT, labels = mm),")");
    print("Độ chính xác trung bình KNN", i, ": ", accuracy_score(y_test, y_pred_KNN)*100);
    # print ("array(\n",confusion_matrix(y_test, y_pred_KNN, labels = mm),")");
    print("Độ chính xác trung bình Bayes", i, ": ", accuracy_score(y_test, y_pred_Bayes)*100);
    # print ("array(\n",confusion_matrix(y_test, y_pred_Bayes, labels = mm),")");














#Nghi thuc k-fold
# KNN = KNeighborsClassifier(n_neighbors=10);
# Bayes = GaussianNB();
# kf = KFold(n_splits=100, shuffle = True, random_state = 6000);
# acc_DT = 0;
# acc_KNN = 0;
# acc_Bayes = 0;
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index];
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index];
#     #DT
#     DT_gini.fit(X_train, y_train);
#     y_pred_DT = DT_gini.predict(X_test);
#     acc_DT += accuracy_score(y_test, y_pred_DT);
#     #KNN
#     KNN.fit(X_train, y_train);
#     y_pred_KNN = KNN.predict(X_test);
#     acc_KNN += accuracy_score(y_test, y_pred_KNN);
#     #Bayes
#     Bayes.fit(X_train, y_train);
#     y_pred_Bayes = Bayes.predict(X_test);
#     acc_Bayes += accuracy_score(y_test, y_pred_Bayes);
# print("Độ chính xác trung bình DT: ", acc_DT/100*100);
# print("Độ chính xác trung bình KNN: ", acc_KNN/100*100);
# print("Độ chính xác trung bình Bayes: ", acc_Bayes/100*100);




