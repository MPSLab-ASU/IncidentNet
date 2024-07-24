import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

def normalize_data(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma

def data_validation(traffic_dataset_path: str,real_traffic: list[int]):
    """
    Performs KS test for simulated traffic data and real world data
    """
    
    df = pd.read_csv(traffic_dataset_path)

    df_day1 = df[df["step"]<86400]
    df_day2 = df[(df["step"]>86400) & (df["step"]<172800)]
    df_day3 = df[df["step"]>172800]

    df_day1_mill = df_day1[df_day1["identified_edge"] == "533573776#0"][["step","time_of_day","traffic_count"]]
    df_day2_mill = df_day2[df_day2["identified_edge"] == "533573776#0"][["step","time_of_day","traffic_count"]]
    df_day3_mill = df_day3[df_day3["identified_edge"] == "533573776#0"][["step","time_of_day","traffic_count"]]

    df_day1_mill['rolling_sum'] = df_day1_mill['traffic_count'].rolling(window=900, min_periods=1).sum()
    df_day2_mill['rolling_sum'] = df_day2_mill['traffic_count'].rolling(window=900, min_periods=1).sum()
    df_day3_mill['rolling_sum'] = df_day3_mill['traffic_count'].rolling(window=900, min_periods=1).sum()

    df_day1_mill_15min = df_day1_mill.iloc[899::900]
    simulated_traffic = df_day1_mill_15min["rolling_sum"].to_list()

    real_traffic_normalized = normalize_data(np.array(real_traffic))
    simulated_traffic_normalized = normalize_data(np.array(simulated_traffic))

    # Repeating Kolmogorov-Smirnov Test with normalized data
    ks_stat, ks_p_value = stats.ks_2samp(real_traffic_normalized, simulated_traffic_normalized)
    print(f"KS Statistic : {ks_stat}, P-value: {ks_p_value}")

    # Calculating Mean Absolute Error (MAE) on normalized data
    mae_normalized = np.mean(np.abs(real_traffic_normalized - simulated_traffic_normalized))
    print(f"Mean Absolute Error (MAE) on : {mae_normalized}")

    return ks_stat,ks_p_value,mae_normalized

