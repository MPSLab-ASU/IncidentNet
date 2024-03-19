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
    

def model_fit(file_path,experiment):
    
    print("Reading csv")
    df = pd.read_csv(file_path)
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
    model.fit(X_train, y_train,sample_weight=sample_weights)
    y_pred = model.predict(X_test)
    # print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("precision: %.2f%%" % (precision * 100.0))
    print("f1: %.2f%%" % (f1 * 100.0))
    print("recall: %.2f%%" % (recall * 100.0))
    
    print("*****************************************************************")
    model_path = f"/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/saved_models/V{experiment}"
    
    os.makedirs(model_path,exist_ok=True)
    pickle.dump(model, open(f"{model_path}/xg_model.pkl", "wb"))
    
    with open(f"{model_path}/logger.txt",'w') as f:
        
        f.write(f"Processed dataset {file_path}\n")
        f.write("Accuracy: %.2f%% \n" % (accuracy * 100.0))
        f.write("precision: %.2f%% \n" % (precision * 100.0))
        f.write("f1: %.2f%% \n" % (f1 * 100.0))
        f.write("recall: %.2f%% \n" % (recall * 100.0))
    
    
directory = "/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/processed_datasets"
files = [os.path.join(directory, f) for f in sorted(os.listdir(directory))]

# files=["/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/processed_datasets/2024-2-16_1915hours_7jun_300_win_300twin.csv"]
experiment = 2
for path in files:
    print(path)
    model_fit(path,experiment)
    experiment+=1