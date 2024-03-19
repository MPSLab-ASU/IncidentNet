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
import os
import pickle
import numpy as np

def callme(prediction_rate,X,Y,xgb_model_loaded,junction):
    
    X= X.values
    Y= Y.values
    y_pred = xgb_model_loaded.predict(X)
    y_pred = list(y_pred)
    y_test = list(Y)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='macro')
    f1 = f1_score(y_test, y_pred,average='macro')
    recall = recall_score(y_test, y_pred,average='macro')
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("precision: %.2f%%" % (precision * 100.0))
    print("f1: %.2f%%" % (f1 * 100.0))
    print("recall: %.2f%%" % (recall * 100.0))

        

    row = {
             "junctions" : junction,
            "Accuracy" : accuracy, 
           "precision": precision,
           "f1":f1,
        
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
        

    df_traffic_data.to_csv(f"/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/evaluation_proccessed/proc_{NUM_OF_JUNCTIONS}_{ROLLING_WINDOW_LENGTH}_{TRAVEL_TIME_WINDOW}.csv")  
    df = df_traffic_data[600:]
    df = df[df["accident_label"] == True]
    df['label'] = None
    
    road_name_edge_id = {
    1:{"934465920","5614812#0","889439250","436794672#0","1078715158"},
    2:{"532215357#0","436794668#0","436794668#7","436794677#0","436794673#0"},
    3:{"436791116","436791119#0","436791122#0","436791121#0","436791111"},
    4:{"1070423862#0","5602753#1","5602753#2","436790493#0","533573789#0","533573789#2","436790492#0","436790484","512811687#0"},
    5:{"436942365#0","436942382#0","436942369#0","436942367","436942384#0","436942364","436942372","395215600","966303717","436942386#0"},
    6:{"532227836","532227834#0","436789544#0","436789576#0","436789539#0","436789539#2","436789539#7","436789570#0"},
    7:{"-436942361#7","-436942361#3","-436942361#1","-436942358#5","-436942358#3","-436942358#0","345713658#0"},
    8:{"-436789319","395490730","1051025192"},
    9:{"436940273#0","692089619#0","692089616#0","692089616#2","436940272#0","436940271#0","5635238","692089613#0","692089613#2","692089613#6","692089611#0"},
    10:{"512810351#0","436943782","436943780","436943781"},
    11:{"436943721","436943716","436943727#0","406379830#0","436943736","436943731#0","436943728","436943723","436943726","436943720","436943747#0","436943754#0","436943740","436943735","436943729","436943741#0","436943752#0"},
    12:{"-436943756#2","-911576955#2","909831620","911576960","-327757100#6"}

    }
    for category, edges in road_name_edge_id.items():
    # Find rows where incident_edge is in the current set of edges
        mask = df['incident_edge'].isin(edges)
    # Set the appropriate label column to 1 for these rows
        df.loc[mask, 'label'] = category-1  # Adjusted index by -1 for zero-based indexing
        
    Y = df["label"]
    X = df.drop(["incident_edge","incident_start_time","incident_type","accident_id","accident_duration","incident_lane","accident_label","label"],axis=1)


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

models = [(3,'/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/saved_models/V4/localization_xg_model.pkl'),
          (4,'/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/saved_models/V8/localization_xg_model.pkl'),
          (5,'/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/saved_models/V12/localization_xg_model.pkl'),
          (6,'/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/saved_models/V16/localization_xg_model.pkl'),
          (7,'/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/saved_models/V20/localization_xg_model.pkl'),
          (8,'/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/xgboost/saved_models/V24/localization_xg_model.pkl')
          ]

for jun,model in models:
    
    rows = evaluate(jun,model)
    
    results = pd.DataFrame(rows)

    
    results.to_csv(f"localization_results_{jun}.csv")
    