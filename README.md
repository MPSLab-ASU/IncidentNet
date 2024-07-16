# traffic_sense_net

## Steps
- Setup Required Software: SUMO, Netedit, OSM Web Wizard
- Generate Simulation file for the region of interest using OSM Web Wizard
- Open OSM and then select a region , enter the simulation duration.
- Generate the simulation files with the above  parameters set.
- Open Netedit, select all the edges between interested junctions (tested for 9 junctions) and also differentiate between junctions and road between junctions.
- Obtain macroscopic traffic data from US DOT
- Generate traffic equation for the above macroscopic data.
- Using the traffic equation, and setting all simulation parameters. Run the simulation for the required duration to generate raw data.
- Preprocess the raw by running the script.
- Using training scripts , train xgboost and tabnet models.
- Using evaluation scripts , run the evaluation on new datasets generated using the saved models to understand the performance.
