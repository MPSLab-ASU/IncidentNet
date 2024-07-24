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
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
np.set_printoptions (threshold=None, edgeitems=None, linewidth=None, suppress=None)


def standardize(col):
   return (col - col.mean()) / col.std()

def xgboost_incident_class_predict(dataset_path: str):

    print("Reading dataset...")
    data_types = {
        'incident_edge': 'object',  # Replace 'Column_Name1' with the actual column name
        'incident_lane': 'object'  # Replace 'Column_Name2' with the actual column name
    }
    df = pd.read_csv(dataset_path,dtype=data_types)
    
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
    
    X = X.drop(["Unnamed: 0","step"],axis=1)
    X= X.values
    Y= Y.values
    
    print("Splitting into test train")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    model = XGBClassifier()
    print("Model training")
    model.fit(X_train, y_train)
    print("Model predicting")
    y_pred = model.predict(X_test)
    y_pred = list(y_pred)
    y_test = list(y_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("precision: %.2f%%" % (precision * 100.0))
    print("f1: %.2f%%" % (f1 * 100.0))
    print("recall: %.2f%%" % (recall * 100.0))
    
    print("Saving model...")
    pickle.dump(model, open(f"incident_class_classification_xg_model.pkl", "wb"))

    return accuracy,precision,f1,recall


