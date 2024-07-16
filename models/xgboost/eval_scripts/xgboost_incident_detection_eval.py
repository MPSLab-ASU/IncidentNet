import pickle
import pandas as pd

import numpy as np
def callme(prediction_rate,X,Y,xgb_model_loaded,junction):
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
    row = {
             "junctions" : junction,
            "prediction_rate" : prediction_rate, 
           "incident_counter": incident_counter,
           "detected_counter":detected_counter,
           "mttd":arr.mean(),
           "false_alarm_counter":false_alarm_counter,
           "predictions_count":predictions_count
        
    }    
    return row
                # print("False Alarm raised by model.")

def standardize(col):
   return (col - col.mean()) / col.std()


def evaluate(NUM_OF_JUNCTIONS,model_path):
    eval_traffic = "/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/evaluation_datasets/V2_eval_trafficDataset_2024-3-16_1753hours_1296000steps.csv"
    eval_vehicle = "/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/evaluation_datasets/V2_eval_vehicleDataset_2024-3-16_1753hours_1296000steps.csv"
    LENGTH_OF_SIM = 1296000
    ROLLING_WINDOW_LENGTH = 600
    TRAVEL_TIME_WINDOW = 600
    print(f"Number of junctions considered for dataset prep: {NUM_OF_JUNCTIONS}")
    print(f"Rolling window length considered for dataset prep: {ROLLING_WINDOW_LENGTH}")
    print(f"Travel time window length considered for dataset prep: {TRAVEL_TIME_WINDOW}")
        
    dtype = {
        'step' : 'int64',
        'time_of_day' : 'int64',
        'identified_edge' : 'object',
        'junction_mean_speed' : 'float64',
        'traffic_count' : 'float64',
        'traffic_occupancy' : 'float64',
        'vehicles_per_lane_1' : 'int64',
        'vehicles_per_lane_0' : 'int64',
        'lane_mean_speed_0' : 'float64',
        'lane_mean_speed_1' : 'float64',
        'incident_edge': 'object',
        'incident_start_time': 'float64',
        'incident_type': 'object',
        'accident_label': 'bool',
        'accident_id': 'object',
        'accident_duration': 'float64',
        'incident_lane': 'object'
    }
    print("Reading traffic csv")
    df_traffic = pd.read_csv(eval_traffic,dtype=dtype)

    INCOMING_EDGES = ['436794670','436791113#0','436794680#0','436940278','-643913497','436943745#0','351673438','-613687451#1','436943750#0','436943762#0','-436942356#1','436942362#0','436789580#1','436942357','436794669','1051038541#0','-436794679#3','533371302#0','436942374','436790491','-436942381#3']

    OUTGOING_EDGES = ['533573776#0','5607328#0','519448767','436940270','-436943745#2','436943742','436943774','436943743#0','-436943762#2','436942356#0','-436942362#3','436789564#0','531969915#0','436794679#0','-1088637809#1','436942385#0','-436794676#1','436790495','-1033824750','30031286#0','436942381#0']


    sensors = INCOMING_EDGES+OUTGOING_EDGES
    print(len(sensors))
    print("Remivng excess sensors")
    sensors_more = {    1:["533573776#0","436794680#0","256917837#0","436791498#0","436794701#0","-436794701#3","5607328#0","436791113#0"],
        2:["436794679#0","-436794679#3","436942387#0","436942371#0","-1088637809#1","436794670","436942385#0","1051038541#0"],
        3:["436942349#0","-436942349#3","-403117287#3","403117287#0","-436794676#1","436794669","531969915#0","436942357"],
        5:["30031286#0","436790491","436942381#0","-436942381#3","436790495","533371302#0","-1033824750","436942374"],
        6:["422264711#0","422264712#0","-436942362#3","436942362#0","436789564#0","436789580#1","436942356#0","-436942356#1"],
        7:["436940270","-643913497","519448767","436940278","436940284","-597312446","422267251","422267252"],
        8:["436943742","351673438","436943774","-613687451#1","-436943745#2","436943745#0","-436943776#4","436943776#0"],
        9:["406379332","436943746#0","-436943762#2","436943762#0","436943743#0","436943750#0","436943757#0","-436943757#1"]
    }

    excess_sensors = []
    for key in sensors_more.keys():
        
        excess_sensors = excess_sensors + sensors_more[key]
        
    set_diff = set(excess_sensors).difference(set(sensors))
    excess_list = list(set_diff)
    df_traffic_reduced = df_traffic[~df_traffic['identified_edge'].isin(excess_list)]


    if NUM_OF_JUNCTIONS == 3:
        print(f" Number of junctions considered 3")
        incoming_edges = ["436794680#0","436791113#0","-436942381#3","533371302#0","436942374","436790491","436943750#0","436943762#0"]
        outgoing_edges = ["533573776#0","5607328#0","436942381#0","436790495","-1033824750","30031286#0","436943743#0","-436943762#2"]
        
    elif NUM_OF_JUNCTIONS == 4:
        print(f" Number of junctions considered 4")
        incoming_edges = ["436794680#0","436791113#0","-436942381#3","533371302#0","436942374","436790491","436943750#0","-643913497","436940278","436943762#0"]
        outgoing_edges = ["533573776#0","5607328#0","436942381#0","436790495","-1033824750","30031286#0","436943743#0","-436943762#2","436940270","519448767"]
        
    elif NUM_OF_JUNCTIONS == 5:
        print(f" Number of junctions considered 5")
        incoming_edges = ["436794680#0","436791113#0","-436942381#3","533371302#0","436942374","436790491","436943750#0","-643913497","436940278","436943762#0","436794669","436942357"]
        outgoing_edges = ["533573776#0","5607328#0","436942381#0","436790495","-1033824750","30031286#0","436943743#0","-436943762#2","436940270","519448767","-436794676#1","531969915#0"]
        
    elif NUM_OF_JUNCTIONS == 6:
        print(f" Number of junctions considered 6")
        incoming_edges = ["436794680#0","436791113#0","-436942381#3","533371302#0","436942374","436790491","436943750#0","-643913497","436940278","436943762#0","436794669","436942357","-436942356#1","436942362#0","436789580#1"]
        outgoing_edges = ["533573776#0","5607328#0","436942381#0","436790495","-1033824750","30031286#0","436943743#0","-436943762#2","436940270","519448767","-436794676#1","531969915#0","436942356#0","-436942362#3","436789564#0"]
        
    elif NUM_OF_JUNCTIONS == 7:
        print(f" Number of junctions considered 7")
        incoming_edges = ["436794680#0","436791113#0","-436942381#3","533371302#0","436942374","436790491","436943750#0","-643913497","436940278","436943762#0","436794669","436942357","-436942356#1","436942362#0","436789580#1","436794670","1051038541#0","-436794679#3"]
        outgoing_edges = ["533573776#0","5607328#0","436942381#0","436790495","-1033824750","30031286#0","436943743#0","-436943762#2","436940270","519448767","-436794676#1","531969915#0","436942356#0","-436942362#3","436789564#0","-1088637809#1","436942385#0","436794679#0"]
        
    elif NUM_OF_JUNCTIONS == 8:
        print(f" Number of junctions considered 8")
        incoming_edges = ["436794680#0","436791113#0","-436942381#3","533371302#0","436942374","436790491","436943750#0","-643913497","436940278","436943762#0","436794669","436942357","-436942356#1","436942362#0","436789580#1","436794670","1051038541#0","-436794679#3","436943745#0","351673438","-613687451#1"]
        outgoing_edges = ["533573776#0","5607328#0","436942381#0","436790495","-1033824750","30031286#0","436943743#0","-436943762#2","436940270","519448767","-436794676#1","531969915#0","436942356#0","-436942362#3","436789564#0","-1088637809#1","436942385#0","436794679#0","-436943745#2","436943742","436943774"]
        

    else:
        print(f" Number of junctions considered {NUM_OF_JUNCTIONS}")
        print("Entered too many sensors resorting to 8 sensors")
        incoming_edges = ["436794680#0","436791113#0","-436942381#3","533371302#0","436942374","436790491","436943750#0","-643913497","436940278","436943762#0","436794669","436942357","-436942356#1","436942362#0","436789580#1","436794670","1051038541#0","-436794679#3","436943745#0","351673438","-613687451#1"]
        outgoing_edges = ["533573776#0","5607328#0","436942381#0","436790495","-1033824750","30031286#0","436943743#0","-436943762#2","436940270","519448767","-436794676#1","531969915#0","436942356#0","-436942362#3","436789564#0","-1088637809#1","436942385#0","436794679#0","-436943745#2","436943742","436943774"]
        
        
    edges = incoming_edges + outgoing_edges

    count = 0
    df = df_traffic_reduced[df_traffic["identified_edge"] == edges[0] ]
    df = df.reset_index(drop=True)
    df[f"rolling_junction_mean_speed_{count}"] = df['junction_mean_speed'].rolling(ROLLING_WINDOW_LENGTH).mean()
    df[f"rolling_traffic_occupancy_{count}"] = df['traffic_occupancy'].rolling(ROLLING_WINDOW_LENGTH).mean()
    df[f"rolling_traffic_count_{count}"] = df['traffic_count'].rolling(ROLLING_WINDOW_LENGTH).mean()

    Y = df[["step","time_of_day","incident_edge","incident_start_time","incident_type","accident_id","accident_duration","incident_lane","accident_label"]]
    df_data = df[["step",f"rolling_junction_mean_speed_{count}",f"rolling_traffic_count_{count}",f"rolling_traffic_occupancy_{count}"]]
    count+=1

    for index,edge in enumerate(edges):
        
        if index == 0:
            continue
        print(f"Processed {count}")
        df = df_traffic_reduced[df_traffic_reduced["identified_edge"] == edge ]
        df = df.reset_index(drop=True)
        df[f"rolling_junction_mean_speed_{count}"] = df['junction_mean_speed'].rolling(ROLLING_WINDOW_LENGTH).mean()
        df[f"rolling_traffic_count_{count}"] = df['traffic_count'].rolling(ROLLING_WINDOW_LENGTH).mean()
        df[f"rolling_traffic_occupancy_{count}"] = df['traffic_occupancy'].rolling(ROLLING_WINDOW_LENGTH).mean()
        
        df_temp = df[["step",f"rolling_junction_mean_speed_{count}",f"rolling_traffic_count_{count}",f"rolling_traffic_occupancy_{count}"]]
        df_data = pd.merge(left = df_data, right = df_temp , on = "step", how = "inner")

        count+=1
        
        
    df_traffic_data = pd.merge(left = df_data, right = Y , on = "step", how = "inner")

    a = df_traffic_data.shape

    print("Shape so far",a[0],a[1])

    # Dealing with Vehicle data
    dtype = {
    'step':'int64',
    'time_of_day':'int64',
    'car_id':'object',
    'identified_edge':'object',
    'identified_lane':'float64',
    'junction_mean_speed':'float64',
    'vehicle_speed':'float64',
    'vehicle_acceleration':'float64'
    }
    print("Reading vehicle csv")
    df_vehicle = pd.read_csv(eval_vehicle,dtype=dtype)


    df_vehicle = df_vehicle[~df_vehicle['identified_edge'].isin(excess_list)]

    df_vehicle = df_vehicle.drop(["junction"],axis=1)

    # sensor details

    sensors = {

        1: "5607328#0",
        2:"436791113#0",
        3:"533573776#0",
        4:"436794680#0",
        5:"436794670",
        6:"436942385#0",
        7:"1051038541#0",
        8:"436794679#0",
        9:"-436794679#3",
        10:"-1088637809#1",
        11:"436794669",
        12:"531969915#0",
        13:"436942357",
        14:"-436794676#1",
        15:"436790495",
        16:"533371302#0",
        17:"-1033824750",
        18:"436942374",
        19:"30031286#0",
        20:"436790491",
        21:"436942381#0",
        22:"-436942381#3",
        23:"436789580#1",
        24:"436942356#0",
        25:"-436942356#1",
        26:"-436942362#3",
        27:"436942362#0",
        28:"436789564#0",
        29:"436940278",
        30:"436940270",
        31:"-643913497",
        32:"519448767",
        33:"436943745#0",
        34:"436943742",
        35:"351673438",
        36:"436943774",
        37:"-613687451#1",
        38:"-436943745#2",
        39:"436943750#0",
        40:"-436943762#2",
        41:"436943762#0",
        42:"436943743#0"

    }

    junctions_sensor_combo ={

        3:[[3,22],[21,4],[1,16],[15,2],[40,20],[19,41],[42,18],[17,39],[3,41],[42,2],[1,39],[40,4]],

        4:[[3,22],[21,4],[1,16],[15,2],[40,20],[19,41],[42,18],[17,39],[3,41],[42,31],[30,39],[40,4],[32,2],[1,29],[32,16],[15,29],[30,18],[17,31]],
        
        5:[[3,11],[14,4],[30,39],[42,31],[32,2],[1,29],[40,13],[12,41],[3,22],[14,22],[21,4],[21,11],[1,16],[15,2],[19,13],[12,20],[32,16],[15,29],[30,18],[17,31],[17,39],[42,18],[19,41],[40,20]],

        6:[[3,11],[14,4],[30,39],[42,31],[32,2],[1,29],[26,13],[12,27],[40,25],[24,41],[3,22],[14,22],[21,4],[21,11],[1,16],[15,2],[19,23],[28,20],[32,16],[15,29],[30,18],[17,31],[17,39],[42,18]],

        7:[[3,5],[10,4],[8,11],[14,9],[30,39],[42,31],[32,2],[1,29],[26,13],[12,27],[40,25],[24,41],[6,22],[21,7],[1,16],[15,2],[19,23],[28,20],[32,16],[15,29],[30,18],[17,31],[17,39],[42,18]],

        8:[[3,5],[10,4],[8,11],[14,9],[30,33],[38,31],[34,39],[42,35],[32,2],[1,29],[26,13],[12,27],[40,25],[24,41],[6,22],[21,7],[1,16],[15,2],[19,23],[28,20],[32,16],[15,29],[36,18],[17,37]]

    }

    interested_combo = junctions_sensor_combo[NUM_OF_JUNCTIONS]

    print(len(interested_combo))

    steps = [i for i in range(LENGTH_OF_SIM)]
    data_dd = {"steps":steps}
    dd_dataframe = pd.DataFrame(data_dd)
    dd_dataframe.shape

    count = 1
    for sensor_combination in interested_combo:
        print(count)
        outgoing_sensor = sensors[sensor_combination[0]]
        incoming_sensor = sensors[sensor_combination[1]]
        df_outgoing = df_vehicle[df_vehicle['identified_edge'] == outgoing_sensor]
        df_incoming = df_vehicle[df_vehicle['identified_edge'] == incoming_sensor]
        df_incoming = df_incoming.reset_index(drop=True)
        df_outgoing = df_outgoing.reset_index(drop=True)

        df_outgoing_unique = df_outgoing[df_outgoing["car_id"].notnull()]
        df_incoming_unique = df_incoming[df_incoming["car_id"].notnull()]

        df_outgoing_unique = df_outgoing_unique.groupby('car_id').first().reset_index()
        df_incoming_unique = df_incoming_unique.groupby('car_id').first().reset_index()

        df_outgoing_unique = df_outgoing_unique.sort_values(by = 'step')
        df_incoming_unique = df_incoming_unique.sort_values(by = 'step')
        
        df_merged = pd.merge(left = df_incoming_unique,right = df_outgoing_unique, on='car_id', how='inner')

        df_merged["travel_time"] = df_merged["step_x"] - df_merged["step_y"]

        df_travel_time_data = df_merged[["step_x","travel_time"]]
        df_travel_time_data = df_travel_time_data.groupby("step_x").mean().reset_index()
        
        df_travel_time_data2 = pd.merge(left = df_travel_time_data,right = dd_dataframe, right_on='steps',left_on="step_x", how='right')

        df_travel_time_data2[f"rolling_travel_time_{sensor_combination[0]}_{sensor_combination[1]}"] = df_travel_time_data2["travel_time"].rolling(300,min_periods=1).mean()

        df_travel_time_data2 = df_travel_time_data2.drop(["step_x","travel_time"],axis=1)
        df_traffic_data = pd.merge(left = df_traffic_data, right = df_travel_time_data2 , left_on = "step",right_on = "steps", how = "inner")
        df_traffic_data = df_traffic_data.drop(["steps"],axis=1)
        count+=1
        
        
    df = df_traffic_data[600:]
    Y = df["accident_label"]
    X = df.drop(["incident_edge","incident_start_time","incident_type","accident_id","accident_duration","incident_lane","accident_label"],axis=1)


    X = X.apply(standardize)

    # file_name = "/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/saved_models/V4/xg_model.pkl"
    xgb_model_loaded = pickle.load(open(model_path, "rb"))

    X = X.drop(["step"],axis=1)

    rows = []
    experiments = [30]

    for i in experiments:
        print("Running experiment on ",i)
        rows.append(callme(i,X,Y,xgb_model_loaded,NUM_OF_JUNCTIONS))
        
    return rows

models = [(3,'/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/saved_models/V4/xg_model.pkl'),
          (4,'/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/saved_models/V8/xg_model.pkl'),
          (5,'/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/saved_models/V12/xg_model.pkl'),
          (6,'/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/saved_models/V16/xg_model.pkl'),
          (7,'/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/saved_models/V20/xg_model.pkl'),
          (8,'/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/saved_models/V24/xg_model.pkl')
          ]

for jun,model in models:
    
    rows = evaluate(jun,model)
    
    results = pd.DataFrame(rows)
    
    results["false_alarm_rate"] = 100*results["false_alarm_counter"]/results["predictions_count"]
    results["accuracy"] = 100*results["detected_counter"]/results["incident_counter"]
    
    results.to_csv(f"results_{jun}.csv")
    