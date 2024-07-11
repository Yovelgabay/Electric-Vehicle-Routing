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
visualize_route(points, route, 'Zigzag Route with Checkpoints Visualization', penalties, connections,
                distances_between_points)

# Apply clustering and genetic algorithm
labels, centroids, num_clusters = kmeans_clustering(points)
# visualize_clustering(num_clusters, points, labels, centroids)
assigned_points = assign_route_points_to_centroids(centroids, route)

# plot_centroids_and_route(centroids, route, assigned_points)

# Define starting point and determine its cluster
starting_point_index = 0  # Specify your starting point index
starting_point_cluster = labels[starting_point_index]

# Run the genetic algorithm
best_charging_stations, _ = genetic_algorithm(points_matrix, route, connections, population_size=POPULATION_SIZE,
                                              generations=GENERATIONS, mutation_rate=MUTATION_RATE,
                                              penalties=penalties, ev_capacity=ev_capacity,
                                              distances_between_points=distances_between_points,
                                              max_stagnation=MAX_STAGNATION,
                                              labels=labels, starting_point_cluster=starting_point_cluster)
print("Best Charging Stations:", best_charging_stations)

demonstrate_chosen_route(route, points, best_charging_stations, connections, 'Chosen Route Visualization',
                         distances_between_points)

print_waiting_times(best_charging_stations, points_matrix, labels, starting_point_cluster,
                    AVERAGE_WAITING_TIME, penalties)

print_segment_lengths(route, connections, best_charging_stations, points_matrix)
