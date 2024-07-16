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
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import os
import pickle
import numpy as np
from tqdm import tqdm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def evaluate(prediction_rate,X,Y,xgb_model_loaded,junction):
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
            predicted = xgb_model_loaded.predict([row])[0] 
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
    row = {
             "junctions" : junction,
            "prediction_rate" : prediction_rate, 
           "incident_counter": incident_counter,
           "detected_counter":detected_counter,
           "mttd":arr.mean(),
           "false_alarm_counter":false_alarm_counter,
           "false_alarm_rate": false_alarm_rate,
           "predictions_count":predictions_count
        
    }    
    return row

from pytorch_tabnet.tab_model import TabNetClassifier
model_path = "/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/wild_models/saved_models/tab_net_8_jun.zip"
loaded_clf = TabNetClassifier()
loaded_clf.load_model(model_path)

df = pd.read_csv("/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/processed_datasets/2024-2-16_1915hours_8jun_600_win_600twin.csv")

df_eval =df.drop(["Unnamed: 0", "step"],axis=1)

def standardize(col):
   return (col - col.mean()) / col.std()


# IF detecting incidents
df_eval = df_eval[600:]
df_eval =df_eval.fillna(-1)

Y = df_eval["accident_label"]

X = df_eval.drop(["incident_edge","incident_start_time","incident_type","accident_id","accident_duration","incident_lane","accident_label"],axis=1)
X = X.apply(standardize)

X= X.values
Y= Y.values

X.shape

# Direct Evaluation - but in reality we run prediction every 1 minute and when an incident flag is raised and detected we have a cooling period after incident clearance
y_pred = loaded_clf.predict(X)
y_pred = list(y_pred)
y_test = list(Y)

accuracy = accuracy_score(y_test, y_pred)*100
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy))
print(f"FAR {100-accuracy}")
print("precision: %.2f%%" % (precision * 100.0))
print("f1: %.2f%%" % (f1 * 100.0))
print("recall: %.2f%%" % (recall * 100.0))

junction = 4
print(evaluate(30,X,Y,loaded_clf,junction))

from sklearn.metrics import roc_auc_score

# Assuming you have true labels (y_true) and predicted labels (y_pred)
auc_score = roc_auc_score(y_test, y_pred)
print("auc_score: %.2f%%" % (auc_score * 100.0))

from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (10,7))
sn.heatmap(matrix, annot=True)


matrix