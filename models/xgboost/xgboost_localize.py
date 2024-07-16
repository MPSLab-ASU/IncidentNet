import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os
import pickle
import numpy as np

def standardize(col):
   return (col - col.mean()) / col.std()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
np.set_printoptions (threshold=None, edgeitems=None, linewidth=None, suppress=None)


dataset_path = "/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/processed_datasets/2024-2-16_1915hours_3jun_600_win_600twin.csv"
data_types = {
    'incident_edge': 'object',  # Replace 'Column_Name1' with the actual column name
    'incident_lane': 'object'  # Replace 'Column_Name2' with the actual column name
}

df = pd.read_csv(dataset_path,dtype=data_types)


    
road_name_edge_id = {
1:{"934465920","5614812#0","889439250","436794672#0","1078715158"},
2:{"532215357#0","436794668#0","436794668#7","436794677#0","436794673#0"},
3:{"436791116","436791119#0","436791122#0","436791121#0","436791111"},
4:{"1070423862#0","5602753#1","5602753#2","436790493#0","533573789#0","533573789#2","436790492#0","436790484","512811687#0"},
5:{"436942365#0","436942382#0","436942369#0","436942367","436942384#0","436942364","436942372","395215600","966303717","436942386#0"},
6:{"532227836","532227834#0","436789544#0","436789576#0","436789539#0","436789539#2","436789539#7","436789570#0"},
7:{"-436942361#7","-436942361#3","-436942361#1","-436942358#5","-436942358#3","-436942358#0","345713658#0"},
8:{"-436789319","395490730","1051025192"},
9:{"436940273#0","692089619#0","692089616#0","692089616#2","436940272#0","436940271#0","5635238","692089613#0","692089613#2","692089613#6","692089611#0"},
10:{"512810351#0","436943782","436943780","436943781"},
11:{"436943721","436943716","436943727#0","406379830#0","436943736","436943731#0","436943728","436943723","436943726","436943720","436943747#0","436943754#0","436943740","436943735","436943729","436943741#0","436943752#0"},
12:{"-436943756#2","-911576955#2","909831620","911576960","-327757100#6"}

}



df = df[df["accident_label"] == True]
df['label'] = None

for category, edges in road_name_edge_id.items():
# Find rows where incident_edge is in the current set of edges
    mask = df['incident_edge'].isin(edges)
# Set the appropriate label column to 1 for these rows
    df.loc[mask, 'label'] = category-1  # Adjusted index by -1 for zero-based indexing

Y = df["label"]

X = df.drop(["incident_edge","incident_start_time","incident_type","accident_id","accident_duration","incident_lane","accident_label","label"],axis=1)
X = X.apply(standardize)

X = X.drop(["Unnamed: 0","step"],axis=1)
X= X.values
Y= Y.values

print("Splitting into test train")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = XGBClassifier(objective='multi:softmax', num_class=12, use_label_encoder=False, eval_metric='mlogloss',device = 'cuda')
print("Model training")
model.fit(X_train, y_train)
print("Model predicting")
y_pred = model.predict(X_test)
y_pred = list(y_pred)
y_test = list(y_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred,average='macro')
f1 = f1_score(y_test, y_pred,average='macro')
recall = recall_score(y_test, y_pred,average='macro')
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("precision: %.2f%%" % (precision * 100.0))
print("f1: %.2f%%" % (f1 * 100.0))
print("recall: %.2f%%" % (recall * 100.0))

pickle.dump(model, open("/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/experiment_saved_models/3jun_inci_local/xg_model.pkl", "wb"))

