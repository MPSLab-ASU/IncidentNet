import pandas as pd
df = pd.read_csv("/home/local/ASURITE/speddira/dev/git_repo/traffic_sense_net/city_scale/processed_datasets/incidents_2024-2-16_1915hours_2592000steps.csv")

df.head()

print(f'Total number of incidents: {df.shape[0]}')

df_severe = df[df["incident_type"] == "multi_vehicle_collision"]
df_mild = df[df["incident_type"] == "stalled_vehicle"]

print(f"Total number of severe incidents {df_severe.shape[0]}")
print(f"Total number of mild incidents {df_mild.shape[0]}")

print(f"Total number of incidents {df_severe.shape[0]}")
print(f"Average accident clearence time {df_severe['accident_duration'].mean()}")
print(f"Max accident clearence time {df_severe['accident_duration'].max()}")
print(f"Min accident clearence time {df_severe['accident_duration'].min()}")
print(f"Median accident clearence time {df_severe['accident_duration'].median()}")
print(f"Total number of incidents {df_mild.shape[0]}")
print(f"Average accident clearence time {df_mild['accident_duration'].mean()}")
print(f"Max accident clearence time {df_mild['accident_duration'].max()}")
print(f"Min accident clearence time {df_mild['accident_duration'].min()}")
print(f"Median accident clearence time {df_mild['accident_duration'].median()}")
