import pickle
import pandas as pd
import numpy as np
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

def evalute(X,Y,model):
    
    X= X.values
    Y= Y.values
    y_pred = model.predict(X)
    y_pred = list(y_pred)
    y_test = list(Y)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("precision: %.2f%%" % (precision * 100.0))
    print("f1: %.2f%%" % (f1 * 100.0))
    print("recall: %.2f%%" % (recall * 100.0))

    return accuracy,precision,f1,recall

def standardize(col):
   return (col - col.mean()) / col.std()

def xgboost_incident_class_predict_eval(model_path,dataset_path):
    data_types = {
    'incident_edge': 'object',  # Replace 'Column_Name1' with the actual column name
    'incident_lane': 'object'  # Replace 'Column_Name2' with the actual column name
    }

    df_traffic_data = pd.read_csv(dataset_path,dtype=data_types)

    df = df_traffic_data[600:]
    df = df[df["accident_label"] == True]
    df['label'] = None

    incident_category = ["stalled_vehicle","multi_vehicle_collision"]
    for category in incident_category:
        # Find rows where incident_edge is in the current set of edges
        mask = df['incident_type'] == category
        # Set the appropriate label column to 1 for these rows
        if category ==  "stalled_vehicle":
            df.loc[mask, 'label'] = 0
        else:
            df.loc[mask, 'label'] = 1

    Y = df["label"]
    X = df.drop(["incident_edge","incident_start_time","incident_type","accident_id","accident_duration","incident_lane","accident_label","label"],axis=1)
    X = X.apply(standardize)

    xgb_model_loaded = pickle.load(open(model_path, "rb"))

    X = X.drop(["step"],axis=1)

    return evalute(X,Y,xgb_model_loaded)