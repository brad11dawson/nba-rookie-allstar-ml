import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
#from sklearn.preprocessing import scale

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

data = pd.read_csv("nba_data3.csv")
data = data.fillna(0)
target = data["All_Star"]
#data = data.drop(columns=["Player", "Yrs", "Year_End", "All_Star"])
data = data.drop(columns=["Player", "Yrs", "All_Star"])

data = normalize(data)

data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.4, random_state=219)


#gradient boosted machine
tree = DecisionTreeClassifier(max_depth=4)
tree.fit(data_train, target_train)
#tree.apply(data_test)
print(tree.feature_importances_)
pred = tree.predict(data_test)

#plot_tree(tree, fontsize=16)
#accuracy
acc = accuracy_score(pred, target_test)

#The precision is intuitively the ability of the 
#classifier not to label as positive a sample that is negative.
# tp/ (tp + fp)
prec = precision_score(target_test, pred)

#ability to find all positive samples
# tp / (tp + fn)
rec = recall_score(target_test, pred)
#Confusion matrix
tn, fp, fn, tp = confusion_matrix(target_test, pred).ravel()

print(f"accuracy: {acc}")
print(f"precision: {prec}")
print(f"recall: {rec}")
print(f"True Negative: {tn}\nTrue Positive: {tp}\nFalse Negative: {fn}\nFalse Positive: {fp}")

#for col in data.columns:
#       print(col)
#booker = gbc.predict([[19, 76, 2108, 367, 867, 99, 289, 215, 256, 27, 187, 200, 44, 20, 160, 225, 1048, .423, .343, .840, 27.7, 13.8, 2.5, 2.6]])
#print(booker)
