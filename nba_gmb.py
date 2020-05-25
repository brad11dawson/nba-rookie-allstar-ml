import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
import numpy as np
#from sklearn.preprocessing import scale

def normalize(df):
    result = df.copy()
    maxs = []
    mins = []
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        maxs.append(max_value)
        mins.append(min_value)
    return result, maxs, mins

data = pd.read_csv("nba_data3.csv")
data = data.fillna(0)
target = data["All_Star"]
#data = data.drop(columns=["Player", "Yrs", "Year_End", "All_Star"])
data = data.drop(columns=["Player", "Yrs", "All_Star"])

#data, maxs, mins = normalize(data)
#6 gives a good return
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.4, random_state=6)


#gradient boosted machine
#gbc =  GradientBoostingClassifier()
#gbc = GradientBoostingClassifier(n_estimators=2000, learning_rate=.01, max_depth=6, subsample=.4)
gbc = GradientBoostingClassifier(n_estimators = 3000, learning_rate = .01, subsample=.5, max_depth=5)
gbc.fit(data_train, target_train)
pred = gbc.predict(data_test)


#accuracy
acc = accuracy_score(pred, target_test)

#The precision is intuitively the ability of the 
#classifier not to label as positive a sample that is negative.
# tp/ (tp + fp)
prec = precision_score(target_test, pred)

#ability to find all positive samples
# tp / (tp + fn)
rec = recall_score(target_test, pred)

f_score = f1_score(target_test, pred)
#Confusion matrix
tn, fp, fn, tp = confusion_matrix(target_test, pred).ravel()

#print(gbc.feature_importances_)
print(f"accuracy: {acc}")
print(f"precision: {prec}")
print(f"recall: {rec}")
print(f"F Score: {f_score}")
#print(f"True Negative: {tn}\nTrue Positive: {tp}\nFalse Negative: {fn}\nFalse Positive: {fp}")

rookies = pd.read_csv("rookies_2018.csv")
rookies = rookies.fillna(0)
rookie_stats = rookies.drop(columns=["Yrs", "Player"])

#normalize the rookies from the mins and maxs of the original data
#for col, minimum, maximum in zip(rookie_stats.columns, mins, maxs):
#       rookie_stats[col] = (rookie_stats[col] - minimum) / (maximum - minimum)
#result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

predictions_2018 = gbc.predict(rookie_stats)
#print(predictions_2018)

rookie_names = rookies.Player
rookie_allstar = np.column_stack((rookie_names, predictions_2018))
#print(rookie_allstar)
# print the players that are allstars
print(rookie_allstar[rookie_allstar[:,1] == 1])
#for col in data.columns:
#       print(col)
#booker = gbc.predict([[19, 76, 2108, 367, 867, 99, 289, 215, 256, 27, 187, 200, 44, 20, 160, 225, 1048, .423, .343, .840, 27.7, 13.8, 2.5, 2.6]])
#print(booker)
