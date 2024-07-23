import math
from Code.functions import assign_route_points_to_centroids, generate_route_with_checkpoints, \
    calculate_distances_between_points, generate_random_points_and_penalties, get_intersection_points, closest_point
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
distances_between_points = calculate_distances_between_points(route)
points_matrix, penalties = generate_random_points_and_penalties(13, NUM_POINTS, 100, route)

# Extract points from points_matrix
points = np.array([point[1] for point in points_matrix])

starting_point_index = 0

# Visualize initial setup
intersections = get_intersection_points(route, points)
connections = [(idx, closest_point(route, pt)) for idx, pt, _ in intersections]

labels, centroids, num_clusters = kmeans_clustering(points, math.ceil(num_route_points / 3))
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

    # Update points_matrix and penalties by removing the first `values_to_remove` entries
    updated_points_matrix = points_matrix[values_to_remove:]
    updated_penalties = penalties[values_to_remove:]
    updated_points = np.array([point[1] for point in updated_points_matrix])
    updated_distances_between_points = calculate_distances_between_points(updated_route)

    # Apply clustering and genetic algorithm
    starting_point_cluster = labels[starting_point_index]

    assigned_points = assign_route_points_to_centroids(centroids, updated_route)

    # Deduct segment distance from initial_ev_capacity
    if starting_point_index > 0:
        initial_ev_capacity -= distances_between_points[starting_point_index - 1]
        print("initial_ev_capacity", initial_ev_capacity)

    # Run the genetic algorithm
    best_charging_stations, _, generations_data = genetic_algorithm(updated_points_matrix, updated_route,
                                                                    updated_connections,
                                                                    population_size=POPULATION_SIZE,
                                                                    generations=GENERATIONS,
                                                                    mutation_rate=MUTATION_RATE,
                                                                    penalties=updated_penalties,
                                                                    ev_capacity=ev_capacity,
                                                                    initial_ev_capacity=initial_ev_capacity,
                                                                    distances_between_points=updated_distances_between_points,
                                                                    max_stagnation=MAX_STAGNATION, labels=labels,
                                                                    starting_point_cluster=starting_point_cluster)



    if best_charging_stations and connections[best_charging_stations[0]+values_to_remove][1] == starting_point_index:
        # Add the charging station to the final_chromosome
        final_chromosome.append(best_charging_stations[0]+values_to_remove)
        distances_to_route = calculate_distances_of_cs(points_matrix, route)
        distance_to_route = distances_to_route[best_charging_stations[0]+values_to_remove]
        initial_ev_capacity = ev_capacity - distance_to_route

    print("final_chromosome", final_chromosome)
    best_routes.append(
        (updated_route, updated_points, best_charging_stations, updated_connections, updated_penalties,
         updated_distances_between_points, values_to_remove))

# Call the function to visualize all routes
visualize_all_routes(best_routes)
