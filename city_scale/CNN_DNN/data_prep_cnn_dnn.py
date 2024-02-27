import pandas as dd
import numpy as np


def create_features_files(data, labels, window_size, stride):

    count = 0
    arrayX = []
    arrayY = []
    for i in range(0, data.shape[0] - window_size + 1, stride):

        if count%10_000 == 0 and count!=0:
            print(f"Generated {count} file so far ...")
        window_data = data[i:i + window_size]
        window_labels = labels[i:i + window_size]
        
        arrayX.append(window_data)
        arrayY.append(window_labels)
        

        np.save(f"data/dataset3/features/features_{count}.npy",window_data)
        np.save(f"data/dataset3/labels/labels_{count}.npy",window_labels)

        count+=1
        
    return count


TRAFFIC_DATASET = "/home/local/ASUAD/speddira/dev/streaming-data-city-scale-incident-detection/raw_datasets/trafficDataset_2024-2-16_1915hours_2592000steps.csv"
NUM_OF_JUNCTIONS = 5

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
print(f"Reading raw dataset")
df_traffic = dd.read_csv(TRAFFIC_DATASET,dtype=dtype)

df = df_traffic[['step','identified_edge','junction_mean_speed','traffic_count','traffic_occupancy']]

'''
Junction Numbering
-1- - 2 - - 3-
 |    |     |
-4- - 5 - - 6-
 |    |     |
-7- - 8 - - 9-
'''

junction_1 = ["533573776#0","436794680#0","5607328#0","436791113#0"]
junction_2 = ["436794679#0","-436794679#3","-1088637809#1","436794670","436942385#0","1051038541#0"]
junction_3 = ["-436794676#1","436794669","531969915#0","436942357"]
junction_4 = [] #As we are considering atmost 8 junctions
junction_5 = ["30031286#0","436790491","436942381#0","-436942381#3","436790495","533371302#0","-1033824750","436942374"]
junction_6 = ["-436942362#3","436942362#0","436789564#0","436789580#1","436942356#0","-436942356#1"]
junction_7 = ["436940270","-643913497","519448767","436940278"]
junction_8 = ["436943742","351673438","436943774","-613687451#1","-436943745#2","436943745#0"]
junction_9 = ["-436943762#2","436943762#0","436943743#0","436943750#0"]

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



if NUM_OF_JUNCTIONS == 3:
    print(f" Number of junctions considered 3")
    column_order = junction_1 + junction_5 +  junction_9
    
elif NUM_OF_JUNCTIONS == 4:
    print(f" Number of junctions considered 4")
    column_order = junction_1 + junction_5 + junction_7 +junction_9
    
elif NUM_OF_JUNCTIONS == 5:
    print(f" Number of junctions considered 5")
    column_order = junction_1 + junction_3+ junction_5 + junction_7+ junction_9
    
elif NUM_OF_JUNCTIONS == 6:
    print(f" Number of junctions considered 6")
    column_order = junction_1 + junction_3+ junction_5 + junction_6+ junction_7+ junction_9
    
elif NUM_OF_JUNCTIONS == 7:
    print(f" Number of junctions considered 7")
    column_order = junction_1 + junction_2 + junction_3+ junction_5 + junction_6+ junction_7+ junction_9
    
elif NUM_OF_JUNCTIONS == 8:
    print(f" Number of junctions considered 8")
    column_order = junction_1 + junction_2 + junction_3+ junction_5 + junction_6+ junction_7 + junction_8 + junction_9

else:
    print(f" Number of junctions considered {NUM_OF_JUNCTIONS}")
    print("Entered too many sensors resorting to 8 sensors")
    column_order = junction_1 + junction_2 + junction_3+ junction_5 + junction_6+ junction_7 + junction_8 + junction_9
    
print(f"Total sensors {len(column_order)}")

features = ['junction_mean_speed', 'traffic_count', 'traffic_occupancy']
arrays = []

for feature in features:
    # Pivot the DataFrame directly with pandas
    pivoted = df_traffic.pivot(index='step', columns='identified_edge', values=feature)
    pivoted_ordered = pivoted[column_order]
    # Convert to a NumPy array
    arr = pivoted_ordered.to_numpy()
    arrays.append(arr)
    
#Stacking the array

combined_array = np.stack(arrays, axis=-1)  # This will create an array of shape (1, 5, 3)

print("Combined Array Shape is:", combined_array.shape)

# Target column

y_df = df_traffic[['step','incident_edge','accident_label']]

y_df_grouped = y_df.groupby(by=["step"]).first().reset_index()

labels = np.zeros((y_df_grouped.shape[0], 13))

for category, edges in road_name_edge_id.items():
    # Find rows where incident_edge is in the current set of edges
    mask = y_df_grouped['incident_edge'].isin(edges) & y_df_grouped['accident_label']
    # Set the appropriate label column to 1 for these rows
    labels[mask, category - 1] = 1  # Adjusted index by -1 for zero-based indexing
    
mask = ~y_df_grouped['accident_label']
labels[mask, -1] = 1



window_size = 300
stride = 300

print("Generating Dataset")
print(f"Window size used : {window_size}")
print(f"Stride used : {stride}")
print(f"Number of junctions considered : {NUM_OF_JUNCTIONS}")

count_files = create_features_files(combined_array, labels, window_size, stride)

print(f"Successfully {count_files} files created.")