import pickle
import json
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

def evaluate(X,Y,xgb_model_loaded):
    
    X= X.values
    Y= Y.values
    y_pred = xgb_model_loaded.predict(X)
    y_pred = list(y_pred)
    y_test = list(Y)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='macro')
    f1 = f1_score(y_test, y_pred,average='macro')
    recall = recall_score(y_test, y_pred,average='macro')
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("precision: %.2f%%" % (precision * 100.0))
    print("f1: %.2f%%" % (f1 * 100.0))
    print("recall: %.2f%%" % (recall * 100.0))

    return accuracy,precision,f1,recall

def standardize(col):
   return (col - col.mean()) / col.std()

def get_edge_ids():
    """
    Load the id of edges of the experiment region
    """
    data = json.load("simulation_network_ids.json")
    road_name_edge_id = data["road_name_edge_id"]

    return road_name_edge_id

def xgboost_localize_eval(dataset_path,model_path):
    
    df_traffic_data = pd.read_csv(dataset_path)
    df = df_traffic_data[600:]
    df = df[df["accident_label"] == True]
    df['label'] = None
    
    road_name_edge_id = get_edge_ids()

    for category, edges in road_name_edge_id.items():
    # Find rows where incident_edge is in the current set of edges
        mask = df['incident_edge'].isin(edges)
    # Set the appropriate label column to 1 for these rows
        df.loc[mask, 'label'] = category-1  # Adjusted index by -1 for zero-based indexing
        
    Y = df["label"]
    X = df.drop(["incident_edge","incident_start_time","incident_type","accident_id","accident_duration","incident_lane","accident_label","label"],axis=1)


    X = X.apply(standardize)

    # file_name = "/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/saved_models/V4/xg_model.pkl"
    xgb_model_loaded = pickle.load(open(model_path, "rb"))

    X = X.drop(["step"],axis=1)
        
    return evaluate(X,Y,xgb_model_loaded)

    