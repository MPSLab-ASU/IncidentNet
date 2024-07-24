import pandas as pd
import json

def get_sensors_details(junctions):

    """
    Load the id of edges, id of individual sensors and combination of sensor placement of the experiment 
    region for the given number of junctions with sensors
    """
    data = json.load("sensor_placement_ids.json")

    if junctions > 8:
        print("Incorrect value entered for junctions with sensors")
        ids = []
        junctions_sensor_combo = []
    else:
        ids = data["sensor_pacement_ids"][str(junctions)]
        junctions_sensor_combo = data["junctions_sensor_combo"][str(junctions)]

    return ids,data["SENSORS"],junctions_sensor_combo
    
def read_raw_data(traffic_raw_data,vehicle_raw_data):

    '''
    Read raw datasets
    '''

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
    df_traffic_raw = pd.read_csv(traffic_raw_data,dtype=dtype)

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
    df_vehicle = pd.read_csv(vehicle_raw_data,dtype=dtype)

    return df_traffic_raw,df_vehicle


def generate_processed_dataset(traffic_raw_data,vehicle_raw_data,junctions,window_lengths,travel_time_window,sim_duaration):
    
    OUTPUT_PATH = "processed_dataset"

    edges,sensors,junctions_sensor_combo = get_sensors_details(junctions)
    
    df_traffic,df_vehicle = read_raw_data(traffic_raw_data,vehicle_raw_data)
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
    df_traffic_data.to_csv(f"{OUTPUT_PATH}_{junctions}jun_{window_lengths}_win_{travel_time_window}twin.csv")
    print("Finished Successfully")