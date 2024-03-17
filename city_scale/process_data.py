import pandas as pd

TRAFFIC_DATASET_PATH = "/home/local/ASURITE/speddira/dev/archived/traffic_sense_net/city_scale/raw_datasets/trafficDataset_2024-2-16_1915hours_2592000steps.csv"
VEHICLE_DATASET_PATH = "/home/local/ASURITE/speddira/dev/archived/traffic_sense_net/city_scale/raw_datasets/vehicleDataset_2024-2-16_1915hours_2592000steps.csv"
output_path = "processed_datasets/2024-2-16_1915hours"

junctions = [8,7,6,5,4,3]
windows = [(300,300),(600,300),(900,300),(600,600)]
SIM_DURATION = 2_592_000

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
print("Reading Traffic Dataset")
df_traffic_raw = pd.read_csv(TRAFFIC_DATASET_PATH,dtype=dtype)

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
print("Reading vehicle Dataset")
df_vehicle = pd.read_csv(VEHICLE_DATASET_PATH,dtype=dtype)

SENSORS = {

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

def get_edges(junctions):
    
    if junctions == 3:
        print(f" Number of junctions considered 3")
        incoming_edges = ["436794680#0","436791113#0","-436942381#3","533371302#0","436942374","436790491","436943750#0","436943762#0"]
        outgoing_edges = ["533573776#0","5607328#0","436942381#0","436790495","-1033824750","30031286#0","436943743#0","-436943762#2"]
        
    elif junctions == 4:
        print(f" Number of junctions considered 4")
        incoming_edges = ["436794680#0","436791113#0","-436942381#3","533371302#0","436942374","436790491","436943750#0","-643913497","436940278","436943762#0"]
        outgoing_edges = ["533573776#0","5607328#0","436942381#0","436790495","-1033824750","30031286#0","436943743#0","-436943762#2","436940270","519448767"]
        
    elif junctions == 5:
        print(f" Number of junctions considered 5")
        incoming_edges = ["436794680#0","436791113#0","-436942381#3","533371302#0","436942374","436790491","436943750#0","-643913497","436940278","436943762#0","436794669","436942357"]
        outgoing_edges = ["533573776#0","5607328#0","436942381#0","436790495","-1033824750","30031286#0","436943743#0","-436943762#2","436940270","519448767","-436794676#1","531969915#0"]
        
    elif junctions == 6:
        print(f" Number of junctions considered 6")
        incoming_edges = ["436794680#0","436791113#0","-436942381#3","533371302#0","436942374","436790491","436943750#0","-643913497","436940278","436943762#0","436794669","436942357","-436942356#1","436942362#0","436789580#1"]
        outgoing_edges = ["533573776#0","5607328#0","436942381#0","436790495","-1033824750","30031286#0","436943743#0","-436943762#2","436940270","519448767","-436794676#1","531969915#0","436942356#0","-436942362#3","436789564#0"]
        
    elif junctions == 7:
        print(f" Number of junctions considered 7")
        incoming_edges = ["436794680#0","436791113#0","-436942381#3","533371302#0","436942374","436790491","436943750#0","-643913497","436940278","436943762#0","436794669","436942357","-436942356#1","436942362#0","436789580#1","436794670","1051038541#0","-436794679#3"]
        outgoing_edges = ["533573776#0","5607328#0","436942381#0","436790495","-1033824750","30031286#0","436943743#0","-436943762#2","436940270","519448767","-436794676#1","531969915#0","436942356#0","-436942362#3","436789564#0","A-1088637809#1","A436942385#0","A436794679#0"]
        
    elif junctions == 8:
        print(f" Number of junctions considered 8")
        incoming_edges = ["436794680#0","436791113#0","-436942381#3","533371302#0","436942374","436790491","436943750#0","-643913497","436940278","436943762#0","436794669","436942357","-436942356#1","436942362#0","436789580#1","436794670","1051038541#0","-436794679#3","436943745#0","351673438","-613687451#1"]
        outgoing_edges = ["533573776#0","5607328#0","436942381#0","436790495","-1033824750","30031286#0","436943743#0","-436943762#2","436940270","519448767","-436794676#1","531969915#0","436942356#0","-436942362#3","436789564#0","-1088637809#1","436942385#0","436794679#0","-436943745#2","436943742","436943774"]
        

    else:
        print("Incorrect junction value, terminating iteration")
        return [],[]
    
    return incoming_edges,outgoing_edges
    

def generate_processed_dataset(junctions,window_lengths,travel_time_window,df_traffic,sensors,junctions_sensor_combo,sim_duaration,output_path):
    
    incoming_edges,outgoing_edges = get_edges(junctions)
    edges = incoming_edges+outgoing_edges
    
    count = 0
    df = df_traffic[df_traffic["identified_edge"] == edges[0] ]
    df = df.reset_index(drop=True)
    df[f"rolling_junction_mean_speed_{count}"] = df['junction_mean_speed'].rolling(window_lengths).mean()
    df[f"rolling_traffic_occupancy_{count}"] = df['traffic_occupancy'].rolling(window_lengths).mean()
    df[f"rolling_traffic_count_{count}"] = df['traffic_count'].rolling(window_lengths).mean()

    Y = df[["step","time_of_day","incident_edge","incident_start_time","incident_type","accident_id","accident_duration","incident_lane","accident_label"]]
    df_data = df[["step",f"rolling_junction_mean_speed_{count}",f"rolling_traffic_count_{count}",f"rolling_traffic_occupancy_{count}"]]
    count+=1
    
    
    for index,edge in enumerate(edges):
    
        if index == 0:
            continue
        # print(f"Processed {count}")
        df = df_traffic[df_traffic["identified_edge"] == edge ]
        df = df.reset_index(drop=True)
        df[f"rolling_junction_mean_speed_{count}"] = df['junction_mean_speed'].rolling(window_lengths).mean()
        df[f"rolling_traffic_count_{count}"] = df['traffic_count'].rolling(window_lengths).mean()
        df[f"rolling_traffic_occupancy_{count}"] = df['traffic_occupancy'].rolling(window_lengths).mean()
        
        df_temp = df[["step",f"rolling_junction_mean_speed_{count}",f"rolling_traffic_count_{count}",f"rolling_traffic_occupancy_{count}"]]
        df_data = pd.merge(left = df_data, right = df_temp , on = "step", how = "inner")

        count+=1
        
    
    df_traffic_data = pd.merge(left = df_data, right = Y , on = "step", how = "inner")
    
    interested_combo = junctions_sensor_combo[junctions]
    
    steps = [i for i in range(sim_duaration)]
    data_dd = {"steps":steps}
    dd_dataframe = pd.DataFrame(data_dd)
    dd_dataframe.shape
    
    count = 1
    for sensor_combination in interested_combo:
        # print(count)
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

        
        df_travel_time_data2 = pd.merge(left = df_travel_time_data,right = dd_dataframe, right_on='steps',left_on="step_x", how='right')

        df_travel_time_data2[f"rolling_travel_time_{sensor_combination[0]}_{sensor_combination[1]}"] = df_travel_time_data2["travel_time"].rolling(travel_time_window,min_periods=1).mean()

        df_travel_time_data2 = df_travel_time_data2.drop(["step_x","travel_time"],axis=1)
        df_traffic_data = pd.merge(left = df_traffic_data, right = df_travel_time_data2 , left_on = "step",right_on = "steps", how = "inner")
        df_traffic_data = df_traffic_data.drop(["steps"],axis=1)
        count+=1
        
    print("Writing to csv")
    df_traffic_data.to_csv(f"{output_path}_{junctions}jun_{window_lengths}_win_{travel_time_window}twin.csv")
    
for junction in junctions:
    print(f"Generating data for junction {junction}")
    for combo in windows:

        window_lengths,travel_time_window = combo
        
        print(f"\tGenerating data for combo {window_lengths} & {travel_time_window}")
        generate_processed_dataset(junction,window_lengths,travel_time_window,df_traffic_raw,SENSORS,junctions_sensor_combo,SIM_DURATION,output_path)
        
        
print("Finished Successfully")