import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
import pickle
import os

def standardize(col):
   return (col - col.mean()) / col.std()
    

def xgboost_incident_detect_train(dataset_path: str):
    
    print("Reading dataset")
    df = pd.read_csv(dataset_path)
    df = df[299:]
    Y = df["accident_label"]
    X = df.drop(["incident_edge","incident_start_time","incident_type","accident_id","accident_duration","incident_lane","accident_label"],axis=1)
    X = X.apply(standardize)
    
    X = X.drop(["Unnamed: 0","step"],axis=1)
    X= X.values
    Y= Y.values
    
    print("SPlitting into test train")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train #provide your own target name
    )
    
    model = XGBClassifier()
    print("Starting Training")
    model.fit(X_train, y_train,sample_weight=sample_weights)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("precision: %.2f%%" % (precision * 100.0))
    print("f1: %.2f%%" % (f1 * 100.0))
    print("recall: %.2f%%" % (recall * 100.0))
    
    print("Saving model...")
    pickle.dump(model, open(f"xg_model_incident_detect.pkl", "wb"))

    print("Finished successfully")
    return accuracy,precision,f1,recall
    

    
    
