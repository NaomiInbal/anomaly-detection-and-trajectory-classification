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


def calculate_accident_point(last_x, last_y, angle, distance):
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

    return x, y


def generate_route_points(num_points, distance_increment, noisy_points, has_accident, vehicle_id):
    route_points = []
    current_time = int(time.time())

    for i in range(num_points):
        x = y = i * distance_increment
        global_time = current_time + i * 10  # Increase timestamp by 10 seconds for each point
        accident_value = 1 if i in noisy_points else 0  # Set accident column value based on noisy points
        route_points.append((x, y, global_time, accident_value, vehicle_id))  # Include vehicle_id column

    accident_point = random.choice(noisy_points) if has_accident else -1
    for random_index in noisy_points:
        last_x, last_y, _, _, _ = route_points[random_index - 1]  # Ignore vehicle_id column
        angle = random.uniform(0, math.pi / 4)  # Limit to 45-degree angle

        # Calculate the random noisy point with a 10-meter distance
        noisy_x, noisy_y = calculate_next_point(last_x, last_y, angle, distance_increment)
        route_points.insert(random_index, (noisy_x, noisy_y, current_time + random_index * 10, 0, vehicle_id))

        # Remove the next point after the noisy point
        route_points.pop(random_index + 1)

        # Introduce accident in vehicles with 50% probability
        if has_accident and random_index == accident_point:
            accident_x, accident_y = calculate_accident_point(last_x, last_y, angle, distance_increment)
            route_points[accident_point] = (accident_x, accident_y, current_time + accident_point * 10, 1, vehicle_id)
            # print(vehicle_id) #TODO -  delete this line.

    return route_points


def write_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['local_x', 'local_y', 'global_time', 'accident', 'vehicle_id'])
        for row in data:
            csv_writer.writerow(row)

def creating_synthetic_track_database(tracks_amount):
    num_points = 36  # Number of points along the curve
    distance_increment = 10  # Increment in meters between points
    num_noisy_points = 20  # Number of noisy points

    # Choose 7 random indices for noisy points
    noisy_points = random.sample(range(1, num_points - 1), num_noisy_points)

    combined_route_points = []

    vehicle_data = {}  # Dictionary to store vehicle data
    for vehicle_id in range(1, tracks_amount):  # Create 5 different vehicle tracks
        has_accident = random.random() < 0.2  # 50% probability of having an accident
        route_points = generate_route_points(num_points, distance_increment, noisy_points, has_accident,
                                             f'vehicle_{vehicle_id}')
        combined_route_points.extend(route_points)

        vehicle_data[f'vehicle_{vehicle_id}'] = has_accident
    # Check if a vehicle has an accident, then update all rows for that vehicle
    # for vehicle_id, has_accident in vehicle_data.items():
    #     if has_accident:
    #         for i in range(len(combined_route_points)):
    #             if combined_route_points[i][4] == vehicle_id:
    #                 combined_route_points[i] = (combined_route_points[i][0], combined_route_points[i][1],
    #                                            combined_route_points[i][2], 1, vehicle_id)
    write_to_csv('vehicle_tracks.csv', combined_route_points)

if __name__ == '__main__':
    small_database = 1500 # A parameter of the amount of tracks for the small database
    big_database = 29127 # A parameter of the amount of tracks for the big database
    creating_synthetic_track_database(small_database)
    creating_synthetic_track_database(big_database)
    vehicle = "vehicle_1099"
    data_frame = main.import_data()
    drawing_track(data_frame, vehicle)