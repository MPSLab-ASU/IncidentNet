import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import sklearn.metrics as metrics
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
np.set_printoptions (threshold=None, edgeitems=None, linewidth=None, suppress=None)
import json

def standardize(col):
   return (col - col.mean()) / col.std()

def get_edge_ids():
    """
    Load the id of edges of the experiment region
    """
    data = json.load("simulation_network_ids.json")
    road_name_edge_id = data["road_name_edge_id"]

    return road_name_edge_id

def incident_localization_train(dataset_path: str):

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

    print("Splitting into test train")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    clf = TabNetClassifier(n_d=64, n_a=64, momentum=0.3, n_steps=5, optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params = {"gamma": 0.95,
                        "step_size": 20},
        scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15)

    print("Starting Training...")
    clf.fit(
    X_train, y_train,num_workers = 10,max_epochs=80
    )

    saving_path_name = "tab_net_4_jun_localize"
    print("Saving model...")
    clf.save_model(saving_path_name)

    X= X_test
    Y= y_test
    y_pred = clf.predict(X)
    y_pred = list(y_pred)
    y_test = list(Y)

    accuracy = accuracy_score(y_test, y_pred)*100
    precision = precision_score(y_test, y_pred,average = "macro")
    f1 = f1_score(y_test, y_pred,average = "macro")
    recall = recall_score(y_test, y_pred,average = "macro")
    print("Accuracy: %.2f%%" % (accuracy))
    print(f"FAR {100-accuracy}")
    print("precision: %.2f%%" % (precision * 100.0))
    print("f1: %.2f%%" % (f1 * 100.0))
    print("recall: %.2f%%" % (recall * 100.0))

    return accuracy,precision,f1,recall