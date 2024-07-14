import numpy as np
from Code.functions import assign_route_points_to_centroids, generate_route_with_checkpoints, \
    calculate_distances_between_points, \
    generate_random_points_and_penalties, get_intersection_points, closest_point
from visualization import (demonstrate_chosen_route, plot_centroids_and_route, visualize_route,
                           visualize_clustering, print_segment_lengths, print_waiting_times)
from GA import genetic_algorithm
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

# Visualize initial setup
intersections = get_intersection_points(route, points)
connections = [(idx, closest_point(route, pt)) for idx, pt, _ in intersections]

# Define starting point and determine its cluster
starting_point_index = 0  # Specify your starting point index

updated_route = route[starting_point_index:]

updated_connections = [(x, y) for (x, y) in connections if y >= starting_point_index]

values_to_remove = len(connections) - len(updated_connections)
print(values_to_remove)

print("Connections:")
print(connections)
print("Updated Connections:")
print(updated_connections)
# Update points_matrix and penalties by removing the first `values_to_remove` entries
updated_points_matrix = points_matrix[values_to_remove:]
updated_penalties = penalties[values_to_remove:]

# Display the updated results
print("Updated points_matrix:")
for point in updated_points_matrix:
    print(point)
print("Updated penalties:")
print(updated_penalties)
updated_points = np.array([point[1] for point in updated_points_matrix])
updated_distances_between_points = calculate_distances_between_points(updated_route)

visualize_route(updated_points, updated_route, 'Zigzag Route with Checkpoints Visualization',
                updated_penalties, updated_connections, updated_distances_between_points, values_to_remove,
                starting_point_index)

# Apply clustering and genetic algorithm
labels, centroids, num_clusters = kmeans_clustering(points)
starting_point_cluster = labels[starting_point_index]
# visualize_clustering(num_clusters, points, labels, centroids)
assigned_points = assign_route_points_to_centroids(centroids, route)

# plot_centroids_and_route(centroids, route, assigned_points)


# Run the genetic algorithm
best_charging_stations, _ = genetic_algorithm(points_matrix, route, connections, population_size=POPULATION_SIZE,
                                              generations=GENERATIONS, mutation_rate=MUTATION_RATE,
                                              penalties=penalties, ev_capacity=ev_capacity,
                                              distances_between_points=distances_between_points,
                                              max_stagnation=MAX_STAGNATION,
                                              labels=labels, starting_point_cluster=starting_point_cluster)
print("Best Charging Stations:", best_charging_stations)

# demonstrate_chosen_route(route, points, best_charging_stations, connections, 'Chosen Route Visualization',
#                          distances_between_points)

print_waiting_times(best_charging_stations, points_matrix, labels, starting_point_cluster,
                    AVERAGE_WAITING_TIME, penalties)

print_segment_lengths(route, connections, best_charging_stations, points_matrix)
