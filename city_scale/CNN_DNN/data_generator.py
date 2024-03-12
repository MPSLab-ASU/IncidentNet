#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime
import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("traffic_data_preprocessing.log"),
            logging.StreamHandler(),
        ],
    )


def load_traffic_data(traffic_dataset):
    logging.info(f"Loading traffic data from {traffic_dataset}")
    dtype = {
        "step": "int64",
        "time_of_day": "int64",
        "identified_edge": "object",
        "junction_mean_speed": "float64",
        "traffic_count": "float64",
        "traffic_occupancy": "float64",
        "vehicles_per_lane_1": "int64",
        "vehicles_per_lane_0": "int64",
        "lane_mean_speed_0": "float64",
        "lane_mean_speed_1": "float64",
        "incident_edge": "object",
        "incident_start_time": "float64",
        "incident_type": "object",
        "accident_label": "bool",
        "accident_id": "object",
        "accident_duration": "float64",
        "incident_lane": "object",
    }
    df_traffic = pd.read_csv(traffic_dataset, dtype=dtype)
    logging.info("Traffic data loaded successfully")
    return df_traffic


def prepare_data(df_traffic, num_of_junctions):
    logging.info(f"Preparing data for {num_of_junctions} junctions")
    junction_edges = {
        1: [
            "533573776#0",
            "436794680#0",
            "5607328#0",
            "436791113#0",
        ],
        2: [
            "436794679#0",
            "-436794679#3",
            "-1088637809#1",
            "436794670",
            "436942385#0",
            "1051038541#0",
        ],
        3: [
            "-436794676#1",
            "436794669",
            "531969915#0",
            "436942357",
        ],
        4: [],
        5: [
            "30031286#0",
            "436790491",
            "436942381#0",
            "-436942381#3",
            "436790495",
            "533371302#0",
            "-1033824750",
            "436942374",
        ],
        6: [
            "-436942362#3",
            "436942362#0",
            "436789564#0",
            "436789580#1",
            "436942356#0",
            "-436942356#1",
        ],
        7: [
            "436940270",
            "-643913497",
            "519448767",
            "436940278",
        ],
        8: [
            "436943742",
            "351673438",
            "436943774",
            "-613687451#1",
            "-436943745#2",
            "436943745#0",
        ],
        9: [
            "-436943762#2",
            "436943762#0",
            "436943743#0",
            "436943750#0",
        ],
    }

    column_order = []
    for i in range(1, num_of_junctions + 1):
        column_order += junction_edges.get(i, [])

    features = [
        "junction_mean_speed",
        "traffic_count",
        "traffic_occupancy",
        "vehicles_per_lane_1",
        "vehicles_per_lane_0",
        "lane_mean_speed_0",
        "lane_mean_speed_1",
    ]

    arrays = []
    for feature in features:
        pivoted = df_traffic.pivot(
            index="step", columns="identified_edge", values=feature
        )
        pivoted_ordered = pivoted[column_order]
        arr = pivoted_ordered.to_numpy()
        arrays.append(arr)

    combined_array = np.stack(arrays, axis=-1)

    y_df = df_traffic[["step", "incident_edge", "accident_label"]]
    y_df_grouped = y_df.groupby(by=["step"]).first().reset_index()
    labels = np.zeros((y_df_grouped.shape[0], 13))

    road_name_edge_id = {
        1: {
            "934465920",
            "5614812#0",
            "889439250",
            "436794672#0",
            "1078715158",
        },
        2: {
            "532215357#0",
            "436794668#0",
            "436794668#7",
            "436794677#0",
            "436794673#0",
        },
        3: {
            "436791116",
            "436791119#0",
            "436791122#0",
            "436791121#0",
            "436791111",
        },
        4: {
            "1070423862#0",
            "5602753#1",
            "5602753#2",
            "436790493#0",
            "533573789#0",
            "533573789#2",
            "436790492#0",
            "436790484",
            "512811687#0",
        },
        5: {
            "436942365#0",
            "436942382#0",
            "436942369#0",
            "436942367",
            "436942384#0",
            "436942364",
            "436942372",
            "395215600",
            "966303717",
            "436942386#0",
        },
        6: {
            "532227836",
            "532227834#0",
            "436789544#0",
            "436789576#0",
            "436789539#0",
            "436789539#2",
            "436789539#7",
            "436789570#0",
        },
        7: {
            "-436942361#7",
            "-436942361#3",
            "-436942361#1",
            "-436942358#5",
            "-436942358#3",
            "-436942358#0",
            "345713658#0",
        },
        8: {
            "-436789319",
            "395490730",
            "1051025192",
        },
        9: {
            "436940273#0",
            "692089619#0",
            "692089616#0",
            "692089616#2",
            "436940272#0",
            "436940271#0",
            "5635238",
            "692089613#0",
            "692089613#2",
            "692089613#6",
            "692089611#0",
        },
        10: {
            "512810351#0",
            "436943782",
            "436943780",
            "436943781",
        },
        11: {
            "436943721",
            "436943716",
            "436943727#0",
            "406379830#0",
            "436943736",
            "436943731#0",
            "436943728",
            "436943723",
            "436943726",
            "436943720",
            "436943747#0",
            "436943754#0",
            "436943740",
            "436943735",
            "436943729",
            "436943741#0",
            "436943752#0",
        },
        12: {
            "-436943756#2",
            "-911576955#2",
            "909831620",
            "911576960",
            "-327757100#6",
        },
    }

    for category, edges in road_name_edge_id.items():
        mask = (
            y_df_grouped["incident_edge"].isin(edges) & y_df_grouped["accident_label"]
        )
        labels[mask, category - 1] = 1

    mask = ~y_df_grouped["accident_label"]
    labels[mask, -1] = 1
    logging.info("Data preparation completed")

    return combined_array, labels


def create_features_files(data, labels, window_size, stride, dataset_name):
    logging.info(f"Creating feature files for dataset: {dataset_name}")
    count = 0
    for i in range(0, data.shape[0] - window_size + 1, stride):
        if count % 100_000 == 0 and count != 0:
            print(f"Generated {count} file so far ...")
        window_data = data[i : i + window_size]
        window_labels = labels[i : i + window_size]

        if not os.path.exists(f"{dataset_name}/features"):
            os.makedirs(f"{dataset_name}/features")
        if not os.path.exists(f"{dataset_name}/labels"):
            os.makedirs(f"{dataset_name}/labels")

        np.save(f"{dataset_name}/features/features_{count}.npy", window_data)
        np.save(f"{dataset_name}/labels/labels_{count}.npy", window_labels)

        count += 1

    logging.info(f"Feature files created: {count}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Traffic Data Preprocessing")
    parser.add_argument(
        "--traffic_dataset",
        type=str,
        required=True,
        help="Path to the traffic dataset CSV file",
    )
    parser.add_argument(
        "--window_sizes",
        type=int,
        nargs="+",
        default=[300],
        help="List of window sizes for feature extraction",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=[30],
        help="List of strides for feature extraction",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        default="data",
        help="Path to save the processed dataset",
    )

    args = parser.parse_args()

    setup_logging()
    logging.info("Starting traffic data preprocessing")

    df_traffic = load_traffic_data(args.traffic_dataset)

    for num_of_junctions in range(3, 9):
        combined_array, labels = prepare_data(df_traffic, num_of_junctions)
        for window_size in args.window_sizes:
            for stride in args.strides:
                if window_size > stride:
                    logging.info(
                        f"Creating dataset with window size {window_size} and stride {stride}."
                    )
                    dataset_name = f"{args.dataset_path}/dataset_{num_of_junctions}_junctions_window_{window_size}_stride_{stride}"
                    count_files = create_features_files(
                        combined_array, labels, window_size, stride, dataset_name
                    )
                    logging.info(
                        f"Successfully created {count_files} files for {num_of_junctions} junctions with window size {window_size} and stride {stride}."
                    )

                    with open(f"{args.dataset_path}/README.md", "a") as f:
                        f.write(f"Data generated on : {str(datetime.now())} \n")
                        f.write(f"Window size used : {window_size}\n")
                        f.write(f"Stride used : {stride}\n")
                        f.write(f"Number of junctions : {num_of_junctions}\n")

        logging.info("Traffic data preprocessing completed")


if __name__ == "__main__":
    main()
