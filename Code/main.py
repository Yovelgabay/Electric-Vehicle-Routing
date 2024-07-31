import math
from Code.functions import (
    assign_route_points_to_centroids, generate_route_with_checkpoints,
    calculate_route_points_distances, generate_random_charging_stations_and_queueing_time, get_intersection_points,
    closest_point
)
from GA import genetic_algorithm, calculate_distances_of_cs
from kmeans import kmeans_clustering
from parameters import *
from visualization import visualize_route, visualize_all_routes

# Generate route and charging stations
num_route_points = NUM_ROUTE_POINTS
route = generate_route_with_checkpoints(num_route_points, 100, 10, seed=53, checkpoints=CHECK_POINTS)
charging_stations_matrix, queueing_time = generate_random_charging_stations_and_queueing_time(
    13, NUM_POINTS, 100, route
)
route_points_distances = calculate_route_points_distances(route)

# Extract charging stations from charging_stations_matrix
charging_stations = np.array([charging_station[1] for charging_station in charging_stations_matrix])

# Get intersection points and connections
intersections = get_intersection_points(route, charging_stations)
connections = [(idx, closest_point(route, pt)) for idx, pt, _ in intersections]

# Apply K-means clustering
labels, centroids, num_clusters = kmeans_clustering(charging_stations, math.ceil(num_route_points / 3))

# Assign route points to clusters
assigned_points = assign_route_points_to_centroids(centroids, route)

# # Initial visualization of the route
# visualize_route(charging_stations, route, "route", queueing_time, connections=None, distances=None,
#                 points_diff=0, route_diff=0)

# Variables to store best routes and final chromosome
best_routes = []
final_chromosome = []

initial_ev_capacity = EV_CAPACITY

# Iterate through each starting point in the route
for starting_point_index in range(0, len(route)):
    print(f"Starting Point Index: {starting_point_index}")

    # Update route and connections based on the starting point
    updated_route = route[starting_point_index:]
    updated_connections = [(x, y - starting_point_index) for (x, y) in connections if y >= starting_point_index]
    values_to_remove = len(connections) - len(updated_connections)

    # Update charging stations and queueing time by removing the first `values_to_remove` entries
    updated_charging_stations_matrix = charging_stations_matrix[values_to_remove:]
    updated_queueing_time = queueing_time[values_to_remove:]
    updated_charging_stations = np.array([point[1] for point in updated_charging_stations_matrix])
    updated_route_points_distances = calculate_route_points_distances(updated_route)

    # Determine the starting point cluster
    starting_point_cluster = assigned_points[starting_point_index]

    # Deduct segment distance from initial_ev_capacity
    if starting_point_index > 0:
        initial_ev_capacity -= route_points_distances[starting_point_index - 1]
        print("initial_ev_capacity", initial_ev_capacity)

    # Run the genetic algorithm to find the best charging stations
    best_charging_stations, _, generations_data = genetic_algorithm(
        updated_charging_stations_matrix, updated_route, updated_connections,
        population_size=POPULATION_SIZE, generations=GENERATIONS, mutation_rate=MUTATION_RATE,
        queueing_time=updated_queueing_time, ev_capacity=EV_CAPACITY, initial_ev_capacity=initial_ev_capacity,
        route_points_distances=updated_route_points_distances, max_stagnation=MAX_STAGNATION, labels=labels,
        starting_point_cluster=starting_point_cluster
    )

    # Check if the best charging station is valid and update the final chromosome
    if (best_charging_stations and
            connections[best_charging_stations[0] + values_to_remove][1] == starting_point_index):
        final_chromosome.append(best_charging_stations[0] + values_to_remove)
        distances_to_route = calculate_distances_of_cs(charging_stations_matrix, route)
        distance_to_route = distances_to_route[best_charging_stations[0] + values_to_remove]
        initial_ev_capacity = EV_CAPACITY - distance_to_route

    print("final_chromosome", final_chromosome)

    # Store the best routes information
    best_routes.append((
        updated_route, updated_charging_stations, best_charging_stations, updated_connections,
        updated_queueing_time, updated_route_points_distances, values_to_remove
    ))

# Visualize all routes
visualize_all_routes(best_routes, labels, centroids, assigned_points)
