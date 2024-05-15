import pandas as dd

NUM_OF_JUNCTIONS = '3str'
ROLLING_WINDOW_LENGTH = 600
TRAVEL_TIME_WINDOW = 600
TRAFFIC_DATASET = "/home/local/ASURITE/speddira/dev/archived/traffic_sense_net/city_scale/raw_datasets/trafficDataset_2024-2-16_1915hours_2592000steps.csv"
VEHICLE_DATASET = "/home/local/ASURITE/speddira/dev/archived/traffic_sense_net/city_scale/raw_datasets/vehicleDataset_2024-2-16_1915hours_2592000steps.csv"
SIM_DURATION = 2_592_000
generate_incidents_csv = False

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
df_traffic = dd.read_csv(TRAFFIC_DATASET,dtype=dtype)

print("Exeriment on 3 straight sensors")

incoming_edges = ["436794680#0","436791113#0","436794670","1051038541#0","-436794679#3","436794669","436942357"]
outgoing_edges = ["533573776#0","5607328#0","-1088637809#1","436942385#0","436794679#0","-436794676#1","531969915#0"]
edges = incoming_edges + outgoing_edges

count = 0
df = df_traffic[df_traffic["identified_edge"] == edges[0] ]
df = df.reset_index(drop=True)
df[f"rolling_junction_mean_speed_{count}"] = df['junction_mean_speed'].rolling(ROLLING_WINDOW_LENGTH).mean()
df[f"rolling_traffic_occupancy_{count}"] = df['traffic_occupancy'].rolling(ROLLING_WINDOW_LENGTH).mean()
df[f"rolling_traffic_count_{count}"] = df['traffic_count'].rolling(ROLLING_WINDOW_LENGTH).mean()

Y = df[["step","time_of_day","incident_edge","incident_start_time","incident_type","accident_id","accident_duration","incident_lane","accident_label"]]
df_data = df[["step",f"rolling_junction_mean_speed_{count}",f"rolling_traffic_count_{count}",f"rolling_traffic_occupancy_{count}"]]
count+=1

a = df_data.shape

print("Shape so far",a[0],a[1])

print("Running for all edges rolling windows")
for index,edge in enumerate(edges):
    
    if index == 0:
        continue
    print(f"Processed {count}")
    df = df_traffic[df_traffic["identified_edge"] == edge ]
    df = df.reset_index(drop=True)
    df[f"rolling_junction_mean_speed_{count}"] = df['junction_mean_speed'].rolling(ROLLING_WINDOW_LENGTH).mean()
    df[f"rolling_traffic_count_{count}"] = df['traffic_count'].rolling(ROLLING_WINDOW_LENGTH).mean()
    df[f"rolling_traffic_occupancy_{count}"] = df['traffic_occupancy'].rolling(ROLLING_WINDOW_LENGTH).mean()
    
    df_temp = df[["step",f"rolling_junction_mean_speed_{count}",f"rolling_traffic_count_{count}",f"rolling_traffic_occupancy_{count}"]]
    df_data = dd.merge(left = df_data, right = df_temp , on = "step", how = "inner")

    count+=1
    
df_traffic_data = dd.merge(left = df_data, right = Y , on = "step", how = "inner")

print("Reading vehicle data")
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
df_vehicle = dd.read_csv(VEHICLE_DATASET,dtype=dtype)

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
    
    '3str':[[3,5],[10,4],[8,11],[14,9]],

    3:[[3,22],[21,4],[1,16],[15,2],[40,20],[19,41],[42,18],[17,39],[3,41],[42,2],[1,39],[40,4]],

    4:[[3,22],[21,4],[1,16],[15,2],[40,20],[19,41],[42,18],[17,39],[3,41],[42,31],[30,39],[40,4],[32,2],[1,29],[32,16],[15,29],[30,18],[17,31]],
    
    5:[[3,11],[14,4],[30,39],[42,31],[32,2],[1,29],[40,13],[12,41],[3,22],[14,22],[21,4],[21,11],[1,16],[15,2],[19,13],[12,20],[32,16],[15,29],[30,18],[17,31],[17,39],[42,18],[19,41],[40,20]],

    6:[[3,11],[14,4],[30,39],[42,31],[32,2],[1,29],[26,13],[12,27],[40,25],[24,41],[3,22],[14,22],[21,4],[21,11],[1,16],[15,2],[19,23],[28,20],[32,16],[15,29],[30,18],[17,31],[17,39],[42,18]],

    7:[[3,5],[10,4],[8,11],[14,9],[30,39],[42,31],[32,2],[1,29],[26,13],[12,27],[40,25],[24,41],[6,22],[21,7],[1,16],[15,2],[19,23],[28,20],[32,16],[15,29],[30,18],[17,31],[17,39],[42,18]],

    8:[[3,5],[10,4],[8,11],[14,9],[30,33],[38,31],[34,39],[42,35],[32,2],[1,29],[26,13],[12,27],[40,25],[24,41],[6,22],[21,7],[1,16],[15,2],[19,23],[28,20],[32,16],[15,29],[36,18],[17,37]]

}

interested_combo = junctions_sensor_combo[NUM_OF_JUNCTIONS]

print(interested_combo)
print(len(interested_combo))

steps = [i for i in range(SIM_DURATION)]
data_dd = {"steps":steps}
dd_dataframe = dd.DataFrame(data_dd)
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
    
    df_merged = dd.merge(left = df_incoming_unique,right = df_outgoing_unique, on='car_id', how='inner')

    df_merged["travel_time"] = df_merged["step_x"] - df_merged["step_y"]

    df_travel_time_data = df_merged[["step_x","travel_time"]]

    df_travel_time_data = df_travel_time_data.groupby("step_x").mean().reset_index()
    df_travel_time_data2 = dd.merge(left = df_travel_time_data,right = dd_dataframe, right_on='steps',left_on="step_x", how='right')

    df_travel_time_data2[f"rolling_travel_time_{sensor_combination[0]}_{sensor_combination[1]}"] = df_travel_time_data2["travel_time"].rolling(300,min_periods=1).mean()

    df_travel_time_data2 = df_travel_time_data2.drop(["step_x","travel_time"],axis=1)
    df_traffic_data = dd.merge(left = df_traffic_data, right = df_travel_time_data2 , left_on = "step",right_on = "steps", how = "inner")
    df_traffic_data = df_traffic_data.drop(["steps"],axis=1)
    # df_traffic_data = df_traffic_data.persist()
    count+=1
    
print('Writing to csv')
df_traffic_data.to_csv("/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/processed_datasets/processed_3str.csv")

for i in df_traffic_data.columns():
    
    print(i)