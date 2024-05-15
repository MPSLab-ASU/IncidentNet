import pandas as pd

TRAFFIC_DATASET_PATH = "/home/local/ASURITE/kharapan/seattle_trafficDataset_2024-3-27_2348hours_1296000steps.csv"
VEHICLE_DATASET_PATH = "/home/local/ASURITE/kharapan/seattle_vehicleDataset_2024-3-27_2348hours_1296000steps.csv"
output_path = "/home/local/ASURITE/speddira/dev/traffic_sense_net/city_scale/experiments/processed_seattle_dataset"

junctions =  [8]
windows = [(600,600)]
SIM_DURATION = 1296000


print("Reading Traffic Dataset")
df_traffic_raw = pd.read_csv(TRAFFIC_DATASET_PATH)

print(df_traffic_raw["accident_label"].head())

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
df_vehicle = pd.read_csv(VEHICLE_DATASET_PATH)


SENSORS = {
    1: "-428078414#0",
    2: "428306604",
    3: "-689020758#3",
    4: "689020758#3",
    5: "-458056506#2",
    6: "823707606#1",
    7: "428077846#0",
    8: "-978114248#0",
    9: "978114248#0",
    10: "458056506#2",
    11: "-537714299#11",
    12: "428078255#1",
    13: "428078249#11",
    14: "537714299#11",
    15: "332320344#0",
    16: "-332320344#0",
    17: "-754695318#0",
    18: "754695318#0",
    19: "-332320345#1",
    20: "332320345#1",
    21: "332320343#0",
    22: "-332320343#0",
    23: "-1064748689#0",
    24: "-33972511#28",
    25: "33972511#28",
    26: "9113121#0",
    27: "749512554#4",
    28: "1064748689#0",
    29: "-797811783",
    30: "-428312241#0",
    31: "797811782",
    32: "797811783",
    33: "-428085404#0",
    34: "-428085411#0",
    35: "428085411#0",
    36: "1051246043#1",
    37: "673405802#3",
    38: "428085404#1",
    39: "-428085409#0",
    40: "428312236",
    41: "-428312236",
    42: "428085409#0",
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
    
    all_edges = ["-428078414#0",
"428306604",
"-689020758#3",
"689020758#3",
"-458056506#2",
"823707606#1",
"428077846#0",
"-978114248#0",
"978114248#0",
"458056506#2",
"-537714299#11",
"428078255#1",
"428078249#11",
"537714299#11",
"332320344#0",
"-332320344#0",
"-754695318#0",
"754695318#0",
"-332320345#1",
"332320345#1",
"332320343#0",
"-332320343#0",
"-1064748689#0",
"-33972511#28",
"33972511#28",
"9113121#0",
"749512554#4",
"1064748689#0",
"-797811783",
"-428312241#0",
"797811782",
"797811783",
"-428085404#0",
"-428085411#0",
"428085411#0",
"1051246043#1",
"673405802#3",
"428085404#1",
"-428085409#0",
"428312236",
"-428312236",
"428085409#0",]
    
    return all_edges
    

def generate_processed_dataset(junctions,window_lengths,travel_time_window,df_traffic,sensors,junctions_sensor_combo,sim_duaration,output_path):
    

    edges = get_edges(junctions)
    
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
        df_travel_time_data = df_travel_time_data.groupby("step_x").mean().reset_index()
        
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