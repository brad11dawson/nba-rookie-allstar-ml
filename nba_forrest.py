import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
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
#data = data.drop(columns=["Player", "Yrs", "Year_End", "All_Star", "Age", "Field_Goal_Percentage", "3_Point_FG_Percentage", "Free_Throw_Percentage", "Minutes_Played_PG"])
#data = data[["Games_Played", "Age", "Minutes_Played_PG", "Field_Goal_Percentage", "3_Point_FG_Percentage", "Free_Throw_Percentage", "Points_PG", "Rebounds_PG", "Assists_PG"]]

data = normalize(data)

#6 is a good number for random state
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.4, random_state=6)

# random forest
#forest = RandomForestClassifier(n_estimators=500, max_depth=4, random_state=32, max_features="sqrt")
#forest = RandomForestClassifier()
forest = RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_split=4)
forest = forest.fit(data_train, target_train)
pred = forest.predict(data_test)
#accuracy
acc = accuracy_score(pred, target_test)

#The precision is intuitively the ability of the 
#classifier not to label as positive a sample that is negative.
# tp/ (tp + fp)
prec = precision_score(target_test, pred)

#ability to find all positive samples
# tp / (tp + fn)
rec = recall_score(target_test, pred)

#f score
fscore = f1_score(target_test, pred)

#Confusion matrix
tn, fp, fn, tp = confusion_matrix(target_test, pred).ravel()

print(f"accuracy: {acc}")
print(f"precision: {prec}")
print(f"recall: {rec}")
print(f"f score: {fscore}")
print(f"True Negative: {tn}\nTrue Positive: {tp}\nFalse Negative: {fn}\nFalse Positive: {fp}")

#for col in data.columns:
#      print(col)