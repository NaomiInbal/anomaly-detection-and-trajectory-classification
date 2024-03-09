import csv
import math
import random
import time
import main
from matplotlib import pyplot as plt

# This file creating synthetic track database. Each line in the database represents a point on the route, and a collection of points with the same vehicle ID make up a route.
# The database has 3 columns: vehicle_id, accident, global_time, local_y, local_x
# Routes are produced in the reservoir, some of which are normal and some with accidents

# Drawing a track by ID
# params: 1. pointer to data frame. 2.Define the vehicle id for which you want to draw the route
def drawing_track(df, vehicle_id):
    # Filter the dataframe to keep only the data for the selected vehicle
    df_vehicle = df[df['vehicle_id'] == vehicle_id]
    # Extract the local_x and local_y coordinates for the selected vehicle
    x = df_vehicle['local_x'].tolist()
    y = df_vehicle['local_y'].tolist()
    # Plot the route
    plt.plot(x, y)
    plt.xlabel("Local X")
    plt.ylabel("Local Y")
    plt.title(f"Route of Vehicle {vehicle_id}")
    plt.show()

def calculate_velocity(x1, y1, t1, x2, y2, t2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # time_diff = (t2 - t1)
    time_diff = 1 # time_diff = 1 second.
    if time_diff == 0:
        return 0
    else:
        return (((distance) / (time_diff)) * 3.6)

def calculate_acceleration(v1, v2, t1, t2):
    time_diff = (t2 - t1)
    # time_diff = 1
    if time_diff == 0:
        return 0
    else:
        acceleration =  ((v2 - v1) / 3.6) / (time_diff)
        return acceleration


def calculate_next_point(last_x, last_y, angle, distance):
    c = last_x + 10
    d = last_y + 10
    while True:
        x = last_x + distance * math.cos(angle)
        y = last_y + distance * math.sin(angle)
        # Check distance condition
        if math.sqrt((x - c) ** 2 + (y - d) ** 2) <= 5:
            break
        else:
            angle += math.pi / 180  # Increase angle by 1 degree and retry

    return x, y


def calculate_accident_point(last_x, last_y, angle):
    distance = 20  # Increase distance by 5 meters in case of an accident
    c = last_x + 10
    d = last_y + 10
    while True:
        x = last_x + distance * math.cos(angle)
        y = last_y + distance * math.sin(angle)
        # Check distance condition
        if (math.sqrt((x - c) ** 2 + (y - d) ** 2) > 11) and (y > last_y):
            break
        else:
            angle += math.radians(5)  # Increase angle by 5 degree and retry
            distance += 2
    return x, y

def calculate_distance_from_previous_point(x1,x2,y1,y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def generate_route_points(accidents_id,num_points, distance_increment, noisy_points, has_accident, vehicle_id):
    route_points = []
    global_time = int(time.time())
    for i in range(num_points):
        x = y = i * distance_increment
        global_time =  global_time + 1 # Increase timestamp by 1 second for each point
        accident_value = 1 if i in noisy_points else 0  # Set accident column value based on noisy points
        route_points.append((x, y, global_time, accident_value, vehicle_id, 0, 0,0))  # Include vehicle_id column
    accident_point = random.choice(noisy_points) if has_accident else -1
    for index in noisy_points:
        last_x, last_y, _, _, _, _, _,_ = route_points[index - 1]
        angle = random.uniform(0, math.pi / 4)  # Limit to 45-degree angle

        # Calculate the random noisy point with a 10-meter distance
        noisy_x, noisy_y = calculate_next_point(last_x, last_y, angle, distance_increment)
        route_points.insert(index, (noisy_x, noisy_y,global_time, 0, vehicle_id,0,0,0))

        # Remove the next point after the noisy point
        route_points.pop(index + 1)

        # Introduce accident in vehicles with 50% probability
        if has_accident and index == accident_point:
            accidents_id.append(vehicle_id)
            accident_x, accident_y = calculate_accident_point(last_x, last_y, angle)
            route_points[accident_point] = (accident_x, accident_y, global_time, 1, vehicle_id,0,0,0)
            # print(vehicle_id) #TODO -  delete this line.

    return route_points,accidents_id


def write_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['local_x', 'local_y', 'global_time', 'accident', 'vehicle_id','velocity', 'acceleration','distance_from_previous_point'])
        for row in data:
            csv_writer.writerow(row)

def creating_synthetic_track_database(tracks_amount):
    accidents_id = []
    num_points = 36  # Number of points along the curve
    distance_increment = 10  # Increment in meters between points
    num_noisy_points = 20  # Number of noisy points

    # Choose 7 random indices for noisy points
    noisy_points = random.sample(range(1, num_points - 1), num_noisy_points)

    combined_route_points = []

    vehicle_data = {}  # Dictionary to store vehicle data
    for vehicle_id in range(1, tracks_amount):  # Create 5 different vehicle tracks
        has_accident = random.random() < 0.2  # 50% probability of having an accident
        route_points,accidents_id = generate_route_points(accidents_id,num_points, distance_increment, noisy_points, has_accident,
                                             f'vehicle_{vehicle_id}')

        # calculate the velocity ,acceleration and the distance from the previous point for all the points.
        for i in range(1, len(route_points)):
            x1, y1, t1, _, _, _, _, _ = route_points[i - 1]
            x2, y2, t2, _, _, _, _ , _= route_points[i]
            velocity = calculate_velocity(x1, y1, t1, x2, y2, t2)
            route_points[i] = route_points[i][:5] + (velocity, 0 , 0)

        for i in range(1, len(route_points)):
            _, _, t1, _, _, v1, _,_ = route_points[i - 1]
            _, _, t2, _, _, v2, _,_ = route_points[i]
            acceleration = calculate_acceleration(v1, v2, t1, t2)
            route_points[i] = route_points[i][:6] + (acceleration, 0, )

        for i in range(1, len(route_points)):
            x1, y1, _, _, _, _, _,_ = route_points[i - 1]
            x2, y2, _, _, _, _, _ ,_= route_points[i]
            distance_from_previous_point = calculate_distance_from_previous_point(x1, x2, y1, y2)
            route_points[i] = route_points[i][:7] + (distance_from_previous_point,)
        combined_route_points.extend(route_points)

        vehicle_data[f'vehicle_{vehicle_id}'] = has_accident
    # Check if a vehicle has an accident, then update all rows for that vehicle
    for vehicle_id, has_accident in vehicle_data.items():
        if has_accident:
            for i in range(len(combined_route_points)):
                if combined_route_points[i][4] == vehicle_id:
                    combined_route_points[i] = (combined_route_points[i][0], combined_route_points[i][1],
                                               combined_route_points[i][2], 1, vehicle_id, combined_route_points[i][5], combined_route_points[i][6],combined_route_points[i][7])
    if tracks_amount == 1500:
        write_to_csv('small_vehicle_tracks_database.csv', combined_route_points)
    else:
        write_to_csv('big_vehicle_tracks_database.csv', combined_route_points)
    return accidents_id

if __name__ == '__main__':
    small_database = 1500 # A parameter of the amount of tracks for the small database
    big_database = 29127 # A parameter of the amount of tracks for the big database
    accidents_id = creating_synthetic_track_database(small_database)
    # creating_synthetic_track_database(big_database)
    data_frame = main.import_data()
    random_id = random.randint(0,len(accidents_id)-1)  # Increase distance by a random value between 2 and 8 meters in case of an accident
    drawing_track(data_frame, accidents_id[0])
