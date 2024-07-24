import pickle
import pandas as pd
import numpy as np

def evaluate(prediction_rate,X,Y,xgb_model_loaded):
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
    for (index,row),(yrow) in zip(X.iterrows(),Y.to_numpy()):
        
        if counter% prediction_rate == 0:
            
            predictions_count+=1
            predicted = xgb_model_loaded.predict([row.values])[0] 
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
        
        
    arr = np.array(detection_times)
    false_alarm_rate = 100*false_alarm_counter/predictions_count
    arr = np.array(detection_times)
    accuracy = detected_counter/incident_counter * 100
    mttd = arr.mean()

    return accuracy,false_alarm_rate,mttd

def standardize(col):
   return (col - col.mean()) / col.std()


def evaluate(dataset_path,model_path):
        
    data_types = {
    'incident_edge': 'object',  # Replace 'Column_Name1' with the actual column name
    'incident_lane': 'object'  # Replace 'Column_Name2' with the actual column name
    }

    df_traffic_data = pd.read_csv(dataset_path,dtype=data_types)
    df = df_traffic_data[600:]
    Y = df["accident_label"]
    X = df.drop(["incident_edge","incident_start_time","incident_type","accident_id","accident_duration","incident_lane","accident_label"],axis=1)


    X = X.apply(standardize)

    # file_name = "/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/saved_models/V4/xg_model.pkl"
    xgb_model_loaded = pickle.load(open(model_path, "rb"))

    X = X.drop(["step"],axis=1)

    return evaluate(60,X,Y,xgb_model_loaded)