# IncidentNet: Traffic Incident Detection, Localization and Severity Estimation with Sparse Sensing

## 🚀 Setup
- Software Required : Python, SUMO, Netedit, OSM Web Wizard
- Install necessary python modules located in **requirements.txt**


## 📝 Steps

###  Simulation Files Generation
-   The first step is to generate sumo configuration files for the region of interest.
-   Run OSM WebWizard  generate the simulation files by selecting the area/region, entering the simulation duration and traffic scale
-   Copy the simulation files to data_generation directory.

###  Raw Data Generation
-   The next step is to run the simulation and generate raw datasets.
-   **streaming_write_step_based_data_gen.py** runs the simulation, simulates incidents and writes the data to csv. It outputs two csv files.
        > raw_traffic_data.csv stores traffic data measured by sensors near the junctions at every second.
        > raw_vehicle_data.csv stores vehicle data including speed , id , lane observed.
-   Before you run the above, edit the values of simulation duration, accident odds(the likelyhood of incident happening), number of incidents.
-   Using curve fitting method, fit a curve to the real world macroscopic traffic data taken from **department of transportation websites** for the experimental region. Use this equation (scaled down if required) as function to vary simulation scale.
-   This code also requires a **simulation_network_ids.json**. This json contains edge ids of the experimental area taken from the generated network file. Netedit can also be used to retrieve the id values of the edges.
- To create the simulation_network_ids.json. Start with identifying the junctions where you want to measure the traffic. The nested json **junction_sensors** contains junctions numbered starting from 1 as key with array of the ids of the edges at the junction. The nested json **ACCIDENT_EDGE** conatins ids of all edges where an incident can happen. The nested json **road_name_edge_id** contains roads numbered from 1 as keys and its values are ids of all edges contained by that road. These road generally the road betwen two junctions which are used in localization of incident.
- With all the parameters updated and json setup, run streaming_write_step_based_gen.py to generate ther raw data files.

### Data Preprocessing
-   The next step is to process the raw data, combining the traffic metrics and individual vehicle data to create a processed traffic dataset.
-   **process_raw_data.py** is used to process the raw data.
-   It takes file paths for traffic raw data, vehicle raw data, number of junctions considered, rolling window lengths for the traffic data averages.
-   This code requires a sensor_placement_ids.json. It contains 3 nested jsons.
    - **sensor_pacement_ids** contains different sensors count and edge ids of the roads where the sensors will be placed. For example a key of 3 indicates 3 junctions have sensors . We list all the ids of the edges of these 3 junctions.
    - **SENSORS** contains user given id and sumo edge id as a value
    - **junctions_sensor_combo** contains all possible paths a vehicle can take for the given junction/sensor count. Contains observed junctions as key and possible combinations as values.
-   This code generates processed_dataset.csv which can be used to train a machine learning/ deep learning model.
-   **incident_data_analysis.py** code gives a summary of the incidents. It outputs total number of incidents, how many of each type (In this work we used two types of incident : Stalled vehicle and Multi vehicle crash). This code takes the path of process_dataset as the input.

### Model Training
-   In this step we train multiple machine learning and deep learning models for three different tasks:
    - To detect if an incident has happend anywhere on the map by observing the data from the sensors placed at the some of the junctions.
    - To localize the incident and predict on which road the incident has likely occurred.
    - To predict the type of incident: Stalled vehicle or Multi vehicle crash.
-   We used XgBoost and TabNet as the two models.
    - **xgboost_incident_class_predict_train.py**,**xgboost_incident_detect_train.py**,**xgboost_localize_train.py** under training_scripts directory train an xgboost models for the above tasks.
    - Similarly **tabnet_incident_class_predict_train.py**,**tabnet_incident_detect_train.py**,**tabnet_localize_train.py** train a tabnet model for the above tasks.
-   These codes take processed_datasets path and return accuracy,precision,f1,recall as outputs for the test dataset obtained by splitting the test-train splitting the processed_datasets.
-   Running these codes also saves a copy of the models which can be used for the evaluation.
-   To evaluate the models, use the scripts in evalution directory. These codes take model path and dataset as inputs and give accuracy and other metrics as outputs.

### Dataset Validation
-   In this step we validate the simulated dataset with realworld data. We use **Kolmogorov–Smirnov test** to validate it.
-   **data_validation.py** takes two inputs processed traffic dataset path and real world data (taken from Department of Transportation Websites) and returns ks statistic , p value and mean absolute error.