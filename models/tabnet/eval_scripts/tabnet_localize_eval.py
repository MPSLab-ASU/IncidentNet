import pandas as pd
import json
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
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import os
import pickle
import numpy as np
from tqdm import tqdm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def calculate_top_k_accuracy(y_true, y_pred, k=1):
    # Args:
    # y_true: numpy array of shape (n_samples,), true class labels
    # y_pred: numpy array of shape (n_samples, n_classes), predicted probabilities
    # k: int, top k predictions to consider for accuracy calculation
    
    # Get the indices of the top k predictions for each instance
    top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
    
    # Check if the true labels are in the top k predictions
    match_array = np.any(top_k_preds == y_true[:, None], axis=1)
    
    # Calculate accuracy
    top_k_accuracy = np.mean(match_array)
    
    return top_k_accuracy

def standardize(col):
   return (col - col.mean()) / col.std()

def get_edge_ids():
    """
    Load the id of edges of the experiment region
    """
    data = json.load("simulation_network_ids.json")
    road_name_edge_id = data["road_name_edge_id"]

    return road_name_edge_id


def incident_localize_eval(dataset_path,model_path):

    df = pd.read_csv(dataset_path)

    data_types = {
        'incident_edge': 'object',  # Replace 'Column_Name1' with the actual column name
        'incident_lane': 'object'  # Replace 'Column_Name2' with the actual column name
    }

    df = pd.read_csv(dataset_path,dtype=data_types)

    road_name_edge_id = get_edge_ids()


    df = df[df["accident_label"] == True]
    df['label'] = None

    for category, edges in road_name_edge_id.items():
    # Find rows where incident_edge is in the current set of edges
        mask = df['incident_edge'].isin(edges)
    # Set the appropriate label column to 1 for these rows
        df.loc[mask, 'label'] = category-1  # Adjusted index by -1 for zero-based indexing

    df = df.fillna(-1)

    Y = df["label"]

    X = df.drop(["incident_edge","incident_start_time","incident_type","accident_id","accident_duration","incident_lane","accident_label","label"],axis=1)
    X = X.apply(standardize)

    X = X.drop(["Unnamed: 0","step"],axis=1)
    X= X.values
    Y= Y.values

    loaded_clf = TabNetClassifier()
    loaded_clf.load_model(model_path)


    y_pred_proba = loaded_clf.predict_proba(X)
    y_pred_proba_list = list(y_pred_proba)
    y_pred = loaded_clf.predict(X)
    y_test = list(Y)

    print("Top 3 acc:",calculate_top_k_accuracy(Y, y_pred_proba, k=3))
    print("Top 2 acc:",calculate_top_k_accuracy(Y, y_pred_proba, k=2))
    accuracy = accuracy_score(y_test, y_pred)*100
    precision = precision_score(y_test, y_pred,average = "macro")
    f1 = f1_score(y_test, y_pred,average = "macro")
    recall = recall_score(y_test, y_pred,average = "macro")
    print("Accuracy: %.2f%%" % (accuracy))
    print(f"FAR {100-accuracy}")
    print("precision: %.2f%%" % (precision * 100.0))
    print("f1: %.2f%%" % (f1 * 100.0))
    print("recall: %.2f%%" % (recall * 100.0))