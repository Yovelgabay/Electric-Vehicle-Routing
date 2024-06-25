import numpy as np
from Code.functions import assign_route_points_to_centroids, generate_zigzag_route, calculate_distances_between_points, \
    generate_random_points_and_penalties, get_intersection_points, closest_point
from visualization import demonstrate_chosen_route, plot_centroids_and_route, visualize_route
from GA import genetic_algorithm
from kmeans import kmeans_clustering
from parameters import *

ev_capacity = EV_CAPACITY
# Generating data
num_route_points = NUM_ROUTE_POINTS
route = generate_zigzag_route(num_route_points, 100, 10, seed=53)
distances_between_points = calculate_distances_between_points(route)
points_matrix, penalties = generate_random_points_and_penalties(13, NUM_POINTS, 100, route)

# Extract points from points_matrix
points = np.array([point[1] for point in points_matrix])

# Visualize initial setup
intersections = get_intersection_points(route, points)
connections = [(idx, closest_point(route, pt)) for idx, pt, _ in intersections]
visualize_route(points, route, 'Zigzag Route Visualization', penalties, connections, distances_between_points)

# Apply clustering and genetic algorithm
labels, centroids = kmeans_clustering(points)

assigned_points = assign_route_points_to_centroids(centroids, route)

plot_centroids_and_route(centroids, route, assigned_points)

best_route_indices = genetic_algorithm(points_matrix, route, connections, population_size=POPULATION_SIZE,
                                       generations=GENERATIONS, mutation_rate=MUTATION_RATE,
                                       penalties=penalties, ev_capacity=ev_capacity,
                                       distances_between_points=distances_between_points, max_stagnation=MAX_STAGNATION)
print("Best Route:", best_route_indices)

demonstrate_chosen_route(route, points, best_route_indices, connections, 'Chosen Route Visualization',
                         distances_between_points)
