"""
Incident Detection across a city level streets traffic simulation
"""

import traci
import traci.constants as tc
import pandas as pd
import sys
import math
import random
import time
from datetime import datetime
import csv

def eucledian_distance(x1,y1,x2,y2):
    '''
    Calculates the distance between two coordinates
    '''

    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def accident_probability_machine(accident_odds,step,last_incident_step):
    '''
    This function is a probability machine which rolls a die and if the die happens to match a particular number , accident flag is set to true. 
    But before that we ensure multiple incidents do not happen in same day (line no 23) and time difference between two incidents is atleast 
    5000 seconds.
    '''
    
    cars_list = traci.vehicle.getIDList()
    
    if step - last_incident_step < 200: #TODO#7_200:
        return False
    
    for object in cars_list:
        random_number = random.randint(0,accident_odds)
        if random_number== 4:
            return True
    
    return False


def clear_incident(incident_vehicles_list):

    '''
    Removal of accident vehicles
    '''

    print("Incident Cleanup Initiated...")

    for index,incident_vehicle in enumerate(incident_vehicles_list):

        try:
            traci.vehicle.remove(incident_vehicle)
            print(f'Incident Vehicle {incident_vehicle} removed by emergency vehicle')
        except:
            print(f'Incident Vehicle {incident_vehicle} removed by SUMO')

def resume_speed(slowed_cars,clear_flag):

    '''
    Cars normal speed resumed once they pass accident zone
    '''

    slowed_cars_list = slowed_cars.copy()

    for car in slowed_cars:
        
        if clear_flag:
            try:
                traci.vehicle.setSpeed(car,normal_speed)
                continue
            except:
                continue
        try:
            x1,y1 = traci.vehicle.getPosition(car)
            min_distance = eucledian_distance(x1,y1,minX,minY)
            max_distance = eucledian_distance(x1,y1,maxX,maxY)
        except:
            slowed_cars_list.remove(car)
            continue
            
        if incident_type == "stalled_vehicle":
            
            if abs(min_distance) > 60:
                # print("RETURN TO NORMAL SPEED")
                traci.vehicle.setColor(car, (200,200,0))
                traci.vehicle.setSpeed(car,normal_speed)
                slowed_cars_list.remove(car)

        else:
            if abs(min_distance) > 60 and abs(max_distance) > 120:
                # print("RETURN TO NORMAL SPEED")
                traci.vehicle.setColor(car, (200,200,0))
                traci.vehicle.setSpeed(car,normal_speed)
                slowed_cars_list.remove(car)

    return slowed_cars_list



#INCIDENT VARIABLES
ACCIDENT_ODDS = 10#TODO900_000 # HIGHER Value means odds of inicident happening is low
NUMBER_OF_INCIDENTS = 100 # Maximum number of incidents you want to limit to for your dataset.Set this to high value, if you don't want to limit it.
accident_flag = False
accident_id = "None"
incident_type = "None" #Stores types of incident
incident_on_road = False
incident_duration_choice = 0
incident_involved_vehicles = []
incident_edge = "None"
incident_duration_list = []
accident_counter = 0
minX = None
maxX = None
minY = None
maxY = None
reduced_speed = 3 # Speed is calculated in m/sec
normal_speed = 16 
incident_count = 0
incident_lane = -1

#SIMULATION VARIABLES
SIMULATION_DURATION = 1000#1_296_000#2_592_000
DAY_LENGTH = 86_400
INCIDENT_HIGHLIGHT_COLOR = (255,0,0)
SLOWED_CARS_HIGHLIGHT_COLOR = ()
scale = 2
traffic = 2
step = 0 # Time steps in seconds
pi = math.pi
incident_start_timestep = 0
time_capture_list = []
accident_happend_on_same_day = False
slowed_cars = set()

# INCOMING_EDGES = ['436794670','436791113#0','436794680#0','436940278','-643913497','436943745#0','351673438','-613687451#1','436943750#0','436943762#0','-436942356#1','436942362#0','436789580#1','436942357','436794669','1051038541#0','-436794679#3','533371302#0','436942374','436790491','-436942381#3']

# OUTGOING_EDGES = ['533573776#0','5607328#0','519448767','436940270','-436943745#2','436943742','436943774','436943743#0','-436943762#2','436942356#0','-436942362#3','436789564#0','531969915#0','436794679#0','-1088637809#1','436942385#0','-436794676#1','436790495','-1033824750','30031286#0','436942381#0']

tempe_junction_sensors = {                            
            1:["533573776#0","436794680#0","256917837#0","436791498#0","436794701#0","-436794701#3","5607328#0","436791113#0"],
            2:["436794679#0","-436794679#3","436942387#0","436942371#0","-1088637809#1","436794670","436942385#0","1051038541#0"],
            3:["436942349#0","-436942349#3","-403117287#3","403117287#0","-436794676#1","436794669","531969915#0","436942357"],
            5:["30031286#0","436790491","436942381#0","-436942381#3","436790495","533371302#0","-1033824750","436942374"],
            6:["422264711#0","422264712#0","-436942362#3","436942362#0","436789564#0","436789580#1","436942356#0","-436942356#1"],
            7:["436940270","-643913497","519448767","436940278","436940284","-597312446","422267251","422267252"],
            8:["436943742","351673438","436943774","-613687451#1","-436943745#2","436943745#0","-436943776#4","436943776#0"],
            9:["406379332","436943746#0","-436943762#2","436943762#0","436943743#0","436943750#0","436943757#0","-436943757#1"]
            }

chandler_junction_sensors = {                            
        1:["-436910211#2","436910211#0","436910207#0","-436943495#2","-436910149#2","436910149#0"],
        2:["436907102#0","-436907102#1","437381373","436907141","-436910134","436910134","-437381365","437381365"],
        3:["436907103#0","-436907103#3","-436907129#2","436907129#0","-436907104#2","436907104#0","436907132#0","-436907132#1"],
        4:["-436944203#3","436944203#0","436944199#0","-436944199#2","436944204#0","-436944204#2","-436944198#2","436944198#0"],
        5:["436945188","-436945188","437376368#0","-437376368#3","-436944196#2","436944196#0","-436945219#3","436945219#0"],
        6:["436945183#0","-436945183#3","-436945210#2","436945210#0","-436945175#1","436945175#0","436945208#0","-531977178"],
        7:["436891568#0","436891566#0","436891555#0","-436891555#1","436891570#0","436891571#0","-436891558#1","436891558#0"],
        8:["529910126#0","436892238#0","436892257#0","-326364342#3","390874842#0","436891553#0","436892256#0","-436892256#2"],
        9:["436892243#0","436892239#0","-436892250#2","436892250#0","436892245#0","436892240#0","436892251#0","-436892251#1"]
        }

edges_with_sensors = []
junctions = []
for key in chandler_junction_sensors.keys():
    
    edges_with_sensors = edges_with_sensors + chandler_junction_sensors[key]
    for i in range(len(chandler_junction_sensors[key])):
        junctions.append(key)

print(f"Total number of sensors used in this simulation {len(edges_with_sensors)}")
print(f"Length of junctions list  {len(junctions)}")


# Tempe parital area
# ACCIDENT_EDGE = ["934465920",
#                  "5614812#0",
#                  "889439250",
#                  "436794672#0",
#                  "1078715158",
#                  "532215357#0",
#                  "436794668#0",
#                  "436794668#7",
#                  "436794677#0",
#                  "436794673#0",
#                  "1070423862#0",
#                  "5602753#1",
#                  "5602753#2",
#                  "436790493#0",
#                  "533573789#0",
#                  "533573789#2",
#                  "436790492#0",
#                  "436790484",
#                  "512811687#0",
#                  "532227836",
#                  "532227834#0",
#                  "436789544#0",
#                  "436789576#0",
#                  "436789539#0",
#                  "436789539#2",
#                  "436789539#7",
#                  "436789570#0",
#                  "436940273#0",
#                  "692089619#0",
#                  "692089616#0",
#                  "692089616#2",
#                  "436940272#0",
#                  "436940271#0",
#                  "5635238",
#                  "692089613#0",
#                  "692089613#2",
#                  "692089613#6",
#                  "692089611#0",
#                  "436943721",
#                  "436943716",
#                  "436943727#0",
#                  "406379830#0",
#                  "436943736",
#                  "436943731#0",
#                  "436943728",
#                  "436943723",
#                  "436943726",
#                  "436943720",
#                  "436943747#0",
#                  "436943754#0",
#                  "436943740",
#                  "436943735",
#                  "436943729",
#                  "436943741#0",
#                  "436943752#0",
#                  "436791116",
#                  "436791119#0",
#                  "436791122#0",
#                  "436791121#0",
#                  "436791111",
#                  "-436789319",
#                  "395490730",
#                  "1051025192",
#                  "512810351#0",
#                  "436943782",
#                  "436943780",
#                  "436943781",
#                  "436942365#0",
#                  "436942382#0",
#                  "436942369#0",
#                  "436942367",
#                  "436942384#0",
#                  "436942364",
#                  "436942372",
#                  "395215600",
#                  "966303717",
#                  "436942386#0",
#                  "-436943756#2",
#                  "-911576955#2",
#                  "909831620",
#                  "911576960",
#                  "-327757100#6",
#                  "-436942361#7",
#                  "-436942361#3",
#                  "-436942361#1",
#                  "-436942358#5",
#                  "-436942358#3",
#                  "-436942358#0",
#                  "345713658#0"]

#chandler data
ACCIDENT_EDGE=[
            "-402560708#1",
            "-657132290",
            "657132294#0",
            "-5667320#4",
            "-5667320#2",
            "436910209#0",
            "436910139",
            "436910138#0",
            "436910137",
            "436907098#4",
            "436907105#0",
            "436907105#2",
            "676542735#2",
            "436907101",
            "436907100#1",
            "436907099#0",
            "436907099#5",
            "436907099#9",
            "436907099#12",
            "763528418#18",
            "763528418#16",
            "763528418#14",
            "763528418#13",
            "763528418#9",
            "763528418#8",
            "763528418#5",
            "763528418#2",
            "763528418#0",
            "763528417",
            "763528419",
            "402560704#2",
            "402560704#0",
            "131637196#0",
            "972483754#1",
            "972483754#0",
            "436945217",
            "436945218",
            "436945225#4",
            "436945225#3",
            "436945215",
            "436945216",
            "436945222#5",
            "436945222#0",
            "-436945194#3",
            "-436945194#5",
            "-531977179#1",
            "-531977179#2",
            "-436945196#1",
            "-436945211#4",
            "-436945209#2",
            "-382549156#1",
            "-382549156#8",
            "-382549156#13",
            "-80396690#10",
            "-80396690#2",
            "-80396690#0",
            "-436944202",
            "-436944200#1",
            "167557805",
            "436944195#1",
            "436944195#2",
            "436944195#4",
            "436944195#10",
            "436945172#0",
            "436945191#10",
            "436945191#11",
            "436945186#0",
            "436945179#0",
            "436945190#1",
            "436945190#8",
            "436944197#8",
            "436944197#7",
            "436944197#3",
            "436944197#2",
            "436944197#1",
            "436944197#0",
            "436891554#0",
            "529686780",
            "402560703#0",
            "436891559#0",
            "436891560#0",
            "29091769#5",
            "29091769#2",
            "436945214",
            "436892253",
            "529875459",
            "131637199#0",
            "131637197",
            "326364342#6",
            "326364342#5",
            "326364342#4",
            "-436945212#4",
            "-436945212#6",
            "-529886116#1",
            "-436945213#1",
            "-436945213#3",
            "-529886115",
            "-529886114",
            "-529886113",
            "-529886112",
            "-529886111#2",
            "-529886111#3",
            "-436892248#5",
            "436891565",
            "221070278#1",
            "390868417#0",
            "390868417#2",
            "390868417#3",
            "390868417#4",
            "390868417#6",
            "390868417#7",
            "529910125",
            "436892247#0",
            "529910123",
            "529910124#0",
            "529910124#1",
            "436892236#0",
            "529910122#0",
            "529910122#2",
            "529910121",
            "529910120",
            "436892244#0",
            "529910118#1",
            "529910118#0"
        ]



# When only distict cars to be recorded
record_only_distinct_cars = False #Set this variable to be true if you want to record eacch car at an intersection only once
incoming_vehicles = []
outgoing_vehicles = []

# CSV write related variables
time_of_run = datetime.now()
vehicle_file_name = f"raw_datasets/V2_chandler_vehicleDataset_{time_of_run.year}-{time_of_run.month}-{time_of_run.day}_{time_of_run.hour}{time_of_run.minute}hours_{SIMULATION_DURATION}steps.csv"
traffic_file_name = f"raw_datasets/V2_chandler_trafficDataset_{time_of_run.year}-{time_of_run.month}-{time_of_run.day}_{time_of_run.hour}{time_of_run.minute}hours_{SIMULATION_DURATION}steps.csv"
time_file_name = f"time_{time_of_run.year}-{time_of_run.month}-{time_of_run.day}_{time_of_run.hour}{time_of_run.minute}hours_{SIMULATION_DURATION}steps.csv"

traffic_dataset_headers = ['step' ,'time_of_day' ,'identified_edge'  ,'junction_mean_speed'  ,'traffic_count'  ,'traffic_occupancy' ,'vehicles_per_lane_1' ,'vehicles_per_lane_0'  ,'lane_mean_speed_0'  ,'lane_mean_speed_1'  ,'incident_edge'  ,'incident_start_time'  ,'incident_type'  ,'accident_label' ,'accident_id'  ,'accident_duration'  ,"incident_lane", "junction"  ]
vehicle_dataset_headers = ['step','time_of_day','car_id','identified_edge','identified_lane','vehicle_speed','vehicle_acceleration','junction']


with open(traffic_file_name, 'w', newline='') as file1, open(vehicle_file_name, 'w', newline='') as file2:
    trafficWriter = csv.writer(file1)
    vehicleWriter = csv.writer(file2)
    
    # Write headers if necessary
    trafficWriter.writerow(traffic_dataset_headers)
    vehicleWriter.writerow(vehicle_dataset_headers)

    sumo_cmd = ["sumo-gui", "-c", r"D:\Dev\traffic_sense_net\simulation_files\partial_chandler_data\osm.sumocfg"]
    print(f"Scale chosen is {scale}")
    traci.start(sumo_cmd)
    traci.simulation.setScale(scale)
    start_time = time.time()

    while step < SIMULATION_DURATION:
        
        if(step%DAY_LENGTH == 0):
            day = (step//DAY_LENGTH)+1
            print(f"****** Day {day}  Started")
            
        if(step%2000==0):
        # Log current step every 1800 steps
            time_capture_dict = {}
            day = (step//DAY_LENGTH)+1
            print(f'******* Day {day} : {step-DAY_LENGTH*(day-1)} steps have passed.')
            # print(f'Scale set to {traffic}')
            print(f"******* Total {step} steps have passed.")
            # end_time = int(time.time())
            # print(f"")
            # time_capture_dict["step"] = step
            # time_capture_dict["time"] = end_time-start_time
            # time_capture_list.append(time_capture_dict)


        # Scaling traffic
        traffic = 2.2+1*(math.sin(2*pi*step/DAY_LENGTH))
        traci.simulation.setScale(traffic)

        # Accident happen ?
        if(incident_count<NUMBER_OF_INCIDENTS and step > 100):
            accident_flag = False if incident_on_road else accident_probability_machine(ACCIDENT_ODDS,step,incident_start_timestep)

        

        for edge,junction_num in zip(edges_with_sensors,junctions):
            traffic_row = []
            traffic_row.append(step) # index : 0 ; step of simultation
            traffic_row.append(step%DAY_LENGTH) # index : 1 ; time of the day
            traffic_row.append(edge) # index : 2 ; edge being monitored
            traffic_row.append(traci.edge.getLastStepMeanSpeed(edge)) # index : 3 ; mean speed of junction
            traffic_row.append(traci.edge.getLastStepVehicleNumber(edge)) # index : 4 ; vehicle count
            traffic_row.append(traci.edge.getLastStepOccupancy(edge)) # index : 5 ; percentage occupancy
            traffic_row.append(traci.lane.getLastStepVehicleNumber(edge+"_1")) # index : 6 ; lane 1 vehicle count
            traffic_row.append(traci.lane.getLastStepVehicleNumber(edge+"_0")) # index : 7 ; lane 0 vehicle count
            traffic_row.append(traci.lane.getLastStepMeanSpeed(edge+"_0")) # index : 8 ; lane 0 mean speed
            traffic_row.append(traci.lane.getLastStepMeanSpeed(edge+"_1")) # index : 9 ; lane 1 mean speed
            traffic_row.append(incident_edge) # index : 10 ; edge where accident happened
            traffic_row.append(incident_start_timestep) # index : 11 ; step where accident occured
            traffic_row.append(incident_type) # index : 12 ; type of incident
            traffic_row.append(incident_on_road) # index : 13 ; accident label (True or False)
            traffic_row.append(accident_id) # index : 14 ; id of accident
            traffic_row.append(incident_duration_choice) # index : 15 ; duration of incident
            traffic_row.append(incident_lane) # index : 16 ; lane of incident
            traffic_row.append(junction_num) # index : 17 ; junction of edge

            vehicle_ids = traci.edge.getLastStepVehicleIDs(edge)
            for vehicle in vehicle_ids:

                lane_id = traci.vehicle.getLaneID(vehicle)
                lane_index = traci.vehicle.getLaneIndex(vehicle)
                # print(f"Lane index of the car is {lane_index}")
                vehicle_row = []
                vehicle_row.append(step) # index : 0
                vehicle_row.append(step%DAY_LENGTH) # index : 1
                vehicle_row.append(vehicle) # index : 2
                vehicle_row.append(edge) # index : 3
                vehicle_row.append(lane_index) # index : 4
                vehicle_row.append(traci.vehicle.getSpeed(vehicle)) # index : 5
                vehicle_row.append(traci.vehicle.getAcceleration(vehicle)) # index : 6
                vehicle_row.append(junction_num) # index : 17 ; junction of edge
                # vehicle_row.append(incident_edge) # index : 7
                # vehicle_row.append(incident_start_timestep) # index : 8
                # vehicle_row.append(incident_type) # index : 9
                # vehicle_row.append(incident_on_road) # index : 10
                # vehicle_row.append(accident_id) # index : 11
                # vehicle_row.append(incident_duration_choice) # index : 12
                # vehicle_row.append(incident_lane) # index : 13

                vehicleWriter.writerow(vehicle_row)

            if len(vehicle_ids) == 0:
                    # vehicle_row = []
                    # vehicle_row.append(step)
                    # vehicle_row.append(step%DAY_LENGTH)
                    # vehicle_row.append(None)
                    # vehicle_row.append(edge)
                    # vehicle_row.append(None)
                    # vehicle_row.append(None)
                    # vehicle_row.append(None)
                    # # vehicle_row.append(incident_edge)
                    # # vehicle_row.append(incident_start_timestep)
                    # # vehicle_row.append(incident_type)
                    # # vehicle_row.append(incident_on_road)
                    # # vehicle_row.append(accident_id)
                    # # vehicle_row.append(incident_duration_choice)
                    # # vehicle_row.append(incident_lane)

                    # vehicleWriter.writerow(vehicle_row)

                    traffic_row[3] = 0
                    # traffic_row_dict['junction_mean_speed'] = 0
                    traffic_row[4] = 0
                    # traffic_row_dict['traffic_count'] = 0
                    traffic_row[5] = 0
                    # traffic_row_dict['traffic_occupancy'] = 0
                    traffic_row[7] = 0
                    # traffic_row_dict['vehicles_per_lane_0'] = 0
                    traffic_row[6] = 0
                    # traffic_row_dict['vehicles_per_lane_1'] = 0
                    traffic_row[8] = 0
                    # traffic_row_dict['lane_mean_speed_0'] = 0
                    traffic_row[9] = 0
                    # traffic_row_dict['lane_mean_speed_1'] = 0
                    
            trafficWriter.writerow(traffic_row)



                
        if (accident_flag):

            # INCIDENT HAPPENED
            random_edge_selection = random.sample(ACCIDENT_EDGE, 1) #Randomly choose one accident edge
            incident_edge = random_edge_selection[0]
            carsList = traci.edge.getLastStepVehicleIDs(incident_edge) #Get list of cars in that edge

            if len(carsList) != 0:
                accident_counter+=1
                accident_flag = False
                accident_happend_on_same_day = True

                if len(carsList) == 1 or len(carsList) == 2 :
                    incident_on_road = True
                    incident_count+=1
                    incident_start_timestep = step
                    print(f'WARNING INCIDENT no {accident_counter} HAPPENED :Vehicle stalled at {step} on {incident_edge}')
                    incident_duration_choice =  random.randint(900, 2700)
                    random_selection = random.sample(carsList, 1)
                    vehicle1 = random_selection[0]
                    x1,y1 = traci.vehicle.getPosition(vehicle1)

                    minX = x1
                    minY = y1
                    maxX = x1+2
                    maxY = y1

                    print(f"Incident happened at: ({x1},{y1})")
                    incident_lane = traci.vehicle.getLaneID(vehicle1)
                    print(f'Accident vehicle id {vehicle1}')
                    print(f'Incident happened on the lane  {incident_lane}')
                    traci.vehicle.setSpeed(vehicle1,0)
                    traci.vehicle.setParameter(vehicle1, "laneChange", "none")
                    traci.vehicle.setColor(vehicle1, INCIDENT_HIGHLIGHT_COLOR)
                    incident_type = "stalled_vehicle"
                    incident_involved_vehicles.append(vehicle1)

                elif len(carsList) >= 3:
                    
                    incident_on_road = True
                    incident_count+=1
                    incident_start_timestep = step

                    print(f'WARNING INCIDENT no {accident_counter} HAPPENED :Vehicle Collision happened at {step} on {incident_edge}')
                    # incident_duration_list = [2700,2700,2700,3600,3600,3600,1800,5400,3000,3000]#TODO random generator between 45 to 2hrs
                    incident_duration_choice =  random.randint(2700, 7200)
                    random_selection = random.sample(carsList,2)
                    vehicle1 = random_selection[0]
                    vehicle2 = random_selection[1]

                    x1,y1 = traci.vehicle.getPosition(vehicle1)
                    x2,y2 = traci.vehicle.getPosition(vehicle2)

                    minX = min(x1,x2)
                    maxX = max(x1,x2)
                    minY = min(y1,y2)
                    maxY = min(y1,y2)

                    print(f"Accident happened at: ({x1},{y1}) ,({x2},{y2})")
                    i_lane1 = traci.vehicle.getLaneID(vehicle1)
                    i_lane2 = traci.vehicle.getLaneID(vehicle2)
                    print(f'Accident vehicle id {vehicle1}, {vehicle2}')
                    print(f'Accident happened on the lanes  {i_lane1}')
                    print(f'Accident happened on the lanes  {i_lane2}')
                    if i_lane1 == i_lane2:
                        incident_lane = i_lane1
                    else:
                        incident_lane = 2

                    traci.vehicle.setSpeed(vehicle1,0)
                    traci.vehicle.setParameter(vehicle1, "laneChange", "none")
                    traci.vehicle.setSpeed(vehicle2,0)
                    traci.vehicle.setParameter(vehicle2, "laneChange", "none")          
                    traci.vehicle.setColor(vehicle1, INCIDENT_HIGHLIGHT_COLOR)
                    traci.vehicle.setColor(vehicle2, INCIDENT_HIGHLIGHT_COLOR)
                    incident_type = "multi_vehicle_collision"
                    incident_involved_vehicles.append(vehicle1)
                    incident_involved_vehicles.append(vehicle2)

                accident_id = f"acc{step}"
                print(f"{incident_type} Incident Duration Clearance after {incident_duration_choice}")



        if(incident_on_road):

            #People should drive slow and carefully when there is an incident on road
            for car in traci.edge.getLastStepVehicleIDs(incident_edge):
                if car not in incident_involved_vehicles:
                    x1,y1 = traci.vehicle.getPosition(car)
                    min_distance = eucledian_distance(x1,y1,minX,minY)
                    max_distance = eucledian_distance(x1,y1,maxX,maxY)

                    if incident_type == "stalled_vehicle":
                        if abs(min_distance) < 60:
                            traci.vehicle.setColor(car, (180,0,0))
                            # print("Slowed because of less distance")
                            reduced_speed = 2.9
                            traci.vehicle.setSpeed(car,reduced_speed)
                            slowed_cars.add(car)


                    else:
                        if abs(min_distance) < 60 or abs(max_distance) < 60:
                            traci.vehicle.setColor(car, (180,0,0))
                            # print("Slowed because of less distance")
                            reduced_speed = 0.8
                            traci.vehicle.setSpeed(car,reduced_speed)
                            slowed_cars.add(car)

            if len(slowed_cars) != 0:
                slowed_cars = resume_speed(slowed_cars,False)


        if (incident_on_road) and (step - incident_start_timestep > incident_duration_choice):

            '''
            Time to clean up accident
            '''

            clear_incident(incident_involved_vehicles)
            slowed_cars =resume_speed(slowed_cars,True) # Resumes speed of all vehicles to normal
            slowed_cars = set()
            # traci.edge.setMaxSpeed(incident_edge,normal_speed)
            incident_involved_vehicles = []
            incident_on_road = False
            # incident_start_timestep = None
            incident_type = "None"
            accident_id = "None"
            incident_edge = "None"
            incident_lane = -1

        step+=1
        traci.simulationStep()
        

    traci.close()
    sys.stdout.flush()


# time_df = pd.DataFrame(time_capture_list)
end_time = int(time.time())
# time_df.to_csv(f"{time_file_name}.csv")
print(f"Finished successfully in {end_time - start_time}")