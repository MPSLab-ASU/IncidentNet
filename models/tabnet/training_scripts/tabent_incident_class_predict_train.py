import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
import pickle
import numpy as np
np.set_printoptions (threshold=None, edgeitems=None, linewidth=None, suppress=None)



def standardize(col):
   return (col - col.mean()) / col.std()

def incident_classification_train(dataset_path: str):

    """
    TabNet training for incident category classification
    """

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

    df = df.fillna(-1)

    Y = df["label"]

    X = df.drop(["incident_edge","incident_start_time","incident_type","accident_id","accident_duration","incident_lane","accident_label","label"],axis=1)
    X = X.apply(standardize)

    X = X.drop(["Unnamed: 0","step"],axis=1)
    X= X.values
    Y= Y.values

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

    saving_path_name = "tab_net_4_jun_severity"
    print("Saving model...")
    clf.save_model(saving_path_name)

    X= X_test
    Y= y_test
    y_pred = clf.predict(X)
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
    print("Training finished Successfully")

    return accuracy,precision,f1,recall
