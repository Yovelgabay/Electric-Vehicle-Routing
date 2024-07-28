import math
from Code.functions import assign_route_points_to_centroids, generate_route_with_checkpoints, \
    calculate_route_points_distances, generate_random_charging_stations_and_queueing_time, get_intersection_points, \
    closest_point
from visualization import (demonstrate_chosen_route, plot_centroids_and_route, visualize_route,
                           visualize_clustering, print_segment_lengths, print_waiting_times,
                           visualize_best_route_animation, visualize_all_routes)
from GA import genetic_algorithm, calculate_distances_of_cs
from kmeans import kmeans_clustering
from parameters import *

ev_capacity = EV_CAPACITY

# Generating data
num_route_points = NUM_ROUTE_POINTS
route = generate_route_with_checkpoints(num_route_points, 100, 10, seed=53, checkpoints=CHECK_POINTS)
route_points_distances = calculate_route_points_distances(route)
charging_stations_matrix, queueing_time = (
    generate_random_charging_stations_and_queueing_time(13, NUM_POINTS, 100, route))

# Extract charging stations from charging_stations_matrix
charging_stations = np.array([charging_station[1] for charging_station in charging_stations_matrix])

starting_point_index = 0

# Visualize initial setup
intersections = get_intersection_points(route, charging_stations)
connections = [(idx, closest_point(route, pt)) for idx, pt, _ in intersections]

labels, centroids, num_clusters = kmeans_clustering(charging_stations, math.ceil(num_route_points / 3))
starting_point_cluster = labels[starting_point_index]

assigned_points = assign_route_points_to_centroids(centroids, route)

# List to store the best routes for each starting point index
best_routes = []
final_chromosome = []
initial_ev_capacity = ev_capacity

for starting_point_index in range(0, len(route)):
    print(f"Calculating for Starting Point Index: {starting_point_index}")

    updated_route = route[starting_point_index:]
    updated_connections = [(x, y - starting_point_index) for (x, y) in connections if y >= starting_point_index]
    values_to_remove = len(connections) - len(updated_connections)

    # Update charging_stations_matrix and queueing_time by removing the first `values_to_remove` entries
    updated_charging_stations_matrix = charging_stations_matrix[values_to_remove:]
    updated_queueing_time = queueing_time[values_to_remove:]
    updated_charging_stations = np.array([point[1] for point in updated_charging_stations_matrix])
    updated_route_points_distances = calculate_route_points_distances(updated_route)

    # Apply clustering and genetic algorithm
    starting_point_cluster = labels[starting_point_index]

    assigned_points = assign_route_points_to_centroids(centroids, updated_route)

    # Deduct segment distance from initial_ev_capacity
    if starting_point_index > 0:
        initial_ev_capacity -= route_points_distances[starting_point_index - 1]
        print("initial_ev_capacity", initial_ev_capacity)

    # Run the genetic algorithm
    best_charging_stations, _, generations_data = (
        genetic_algorithm(updated_charging_stations_matrix, updated_route,
                          updated_connections,
                          population_size=POPULATION_SIZE,
                          generations=GENERATIONS,
                          mutation_rate=MUTATION_RATE,
                          queueing_time=updated_queueing_time,
                          ev_capacity=ev_capacity,
                          initial_ev_capacity=initial_ev_capacity,
                          route_points_distances=updated_route_points_distances,
                          max_stagnation=MAX_STAGNATION, labels=labels,
                          starting_point_cluster=starting_point_cluster))

    if (best_charging_stations and
            connections[best_charging_stations[0] + values_to_remove][1] == starting_point_index):
        # Add the charging station to the final_chromosome
        final_chromosome.append(best_charging_stations[0] + values_to_remove)
        distances_to_route = calculate_distances_of_cs(charging_stations_matrix, route)
        distance_to_route = distances_to_route[best_charging_stations[0] + values_to_remove]
        initial_ev_capacity = ev_capacity - distance_to_route

    print("final_chromosome", final_chromosome)
    best_routes.append(
        (updated_route, updated_charging_stations, best_charging_stations, updated_connections, updated_queueing_time,
         updated_route_points_distances, values_to_remove))

# Call the function to visualize all routes
visualize_all_routes(best_routes)
