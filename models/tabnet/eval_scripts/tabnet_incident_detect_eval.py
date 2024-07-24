import pandas as pd
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
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import os
import pickle
import numpy as np
from tqdm import tqdm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def standardize(col):
   return (col - col.mean()) / col.std()

def evaluate(prediction_rate,X,Y,model):
    history = [0]  # Assuming no incident at start
    incident_counter = 0
    detected_counter = 0
    new_incident_flag = False
    incident_detected_flag = False
    predictions_count = 0
    i = 0  # Time since last new incident
    false_alarm_counter = 0
    detection_times = []  # To calculate mean time for detection
    counter = 1
    for row,yrow in tqdm(zip(X,Y)):
        
        if counter% prediction_rate == 0:
            
            predictions_count+=1
            predicted = model.predict([row])[0] 
            # print(predicted)
            
            if len(history) > 0:
                previous_label = history[-1]
            else:
                previous_label = 0  # Default to no incident for the first element

            # Update history
            current_label = 1 if yrow else 0
            history.append(current_label)

            # New incident started
            if previous_label == 0 and current_label == 1:
                incident_counter += 1
                new_incident_flag = True
                incident_detected_flag = False  # Reset detection flag for new incident
                i = 0  # Reset timer for new incident
                # print("New incident started")

            # Incident detected
            if new_incident_flag and predicted == 1 and not incident_detected_flag:
                detected_counter += 1
                detection_times.append(i)
                incident_detected_flag = True
                # print(f"Incident Detected by model after {i} timesteps.")

            # Incident finished
            if previous_label == 1 and current_label == 0:
                new_incident_flag = False
                # print("Incident Finished")


            # False alarm raised
            if not new_incident_flag and predicted == 1 and not (1 in history[-1000:]):
                false_alarm_counter += 1
                
        # Increment timer if incident ongoing but not yet detected
        if new_incident_flag and not incident_detected_flag:
            i += 1
        counter+=1
        
    false_alarm_rate = 100*false_alarm_counter/predictions_count
    arr = np.array(detection_times)
    accuracy = detected_counter/incident_counter * 100
    mttd = arr.mean()

    return accuracy,false_alarm_rate,mttd


def tabnet_incident_detect_eval(dataset_path,model_path):
    """
    Entry point
    """

    
    loaded_clf = TabNetClassifier()
    loaded_clf.load_model(model_path)

    data_types = {
            'incident_edge': 'object',  # Replace 'Column_Name1' with the actual column name
            'incident_lane': 'object'  # Replace 'Column_Name2' with the actual column name
        }

    df = pd.read_csv(dataset_path,dtype=data_types)

    df_eval =df.drop(["Unnamed: 0", "step"],axis=1)

    df_eval = df_eval[600:]
    df_eval =df_eval.fillna(-1)

    Y = df["accident_label"]      
    X = df.drop(["Unnamed: 0","step","incident_edge","incident_start_time","incident_type","accident_id","accident_duration","incident_lane","accident_label"],axis=1)
    X = X.apply(standardize)

    X= X.values
    Y= Y.values

    #In reality we run prediction every 1 minute and when an incident flag is raised and detected we have a cooling period after incident clearance

    return evaluate(60,X,Y,loaded_clf)