import copy
import math
import os
import time
from Code.functions import (
    assign_route_points_to_centroids, generate_route_with_checkpoints,
    calculate_route_points_distances, generate_random_charging_stations_and_queueing_time,
    get_intersection_points, closest_point
)

from Code.visualization import visualize_all_routes, visualize_best_route_animation, visualize_clustering
from GA import genetic_algorithm, calculate_distances_of_cs, final_fitness_function
from kmeans import kmeans_clustering
from parameters import *

os.environ['LOKY_MAX_CPU_COUNT'] = '4'


def generate_initial_data():
    """
    Generate initial data for the routing and charging station setup.
    """

    # Generate route with checkpoints
    num_route_points = NUM_ROUTE_POINTS
    route = generate_route_with_checkpoints(
        num_route_points, scale=100, turn_amplitude=10, seed=53, checkpoints=CHECK_POINTS)

    # Generate random charging stations with random queueing times
    charging_stations_matrix, queueing_time = generate_random_charging_stations_and_queueing_time(
        seed=13, num_points=NUM_POINTS, scale=100, route=route
    )

    # Calculate distances between consecutive route points
    route_points_distances = calculate_route_points_distances(route)

    # Extract coordinates of charging stations from the matrix
    charging_stations = np.array([charging_station[1] for charging_station in charging_stations_matrix])

    return route, charging_stations_matrix, queueing_time, route_points_distances, charging_stations


def get_connections(route, charging_stations):
    """
    Find intersections and connections between route points and charging stations.
    """

    # Find intersection points between route and charging stations
    intersections = get_intersection_points(route, charging_stations)

    # For each intersection, find the closest route point and create a connection
    connections = [(idx, closest_point(route, pt)) for idx, pt, _ in intersections]

    return connections


def kmeans_and_assign_clusters(route, charging_stations):
    """
    Apply K-means clustering to charging stations and assign route points to clusters.
    """

    """ 
    Perform K-means clustering on the charging stations.
    The number of clusters is chosen as the ceiling of one-third the length of the route,
    which provides a reasonable number of clusters for effective clustering in the algorithm
    """
    cluster_labels, centroids, num_clusters = kmeans_clustering(charging_stations, math.ceil(len(route) / 3))

    # Assign each route point to the nearest centroid
    assigned_points = assign_route_points_to_centroids(centroids, route)

    # Visualize the clustering
    # visualize_clustering(num_clusters, charging_stations, cluster_labels, centroids)

    return cluster_labels, centroids, assigned_points


def update_route_data(starting_point_index, route, connections, charging_stations_matrix, queueing_time):
    """
    Update route and related data based on a new starting point index.
    """

    # Slice the route starting from the new index
    updated_route = route[starting_point_index:]

    # Update connections to reflect the new starting index
    updated_connections = [(x, y - starting_point_index) for (x, y) in connections if y >= starting_point_index]
    values_to_remove = len(connections) - len(updated_connections)

    # Adjust charging stations matrix and queueing times based on removed values
    updated_charging_stations_matrix = charging_stations_matrix[values_to_remove:]
    updated_queueing_time = queueing_time[values_to_remove:]

    # Extract coordinates of updated charging stations
    updated_charging_stations = np.array([point[1] for point in updated_charging_stations_matrix])

    # Recalculate distances between consecutive route points
    updated_route_points_distances = calculate_route_points_distances(updated_route)

    return (updated_route, updated_connections, updated_charging_stations_matrix, updated_queueing_time,
            updated_charging_stations, updated_route_points_distances, values_to_remove)


def run_genetic_algorithm_for_each_start_point(route, connections, cluster_labels, assigned_points, charging_stations_matrix,
                                               queueing_time, route_points_distances):
    """
    Run the genetic algorithm for each starting point on the route and gather the best routes.
    """

    # Initialize variables to store the best routes and the final chromosome
    best_routes = []
    final_chromosome = []
    initial_ev_capacity = EV_CAPACITY  # Initial EV capacity
    next_population_candidates = []  # List to store cs for inclusion in the next population at the next starting point
    prev_removed_count = 0  # Number of cs removed in the previous step, used to maintain correct indexing

    # Iterate over each point on the route as a potential starting point
    for current_start_index in range(len(route)):

        # Update the route and relevant data based on the current starting point
        updated_route, updated_connections, updated_charging_stations_matrix, updated_queueing_time, \
            updated_charging_stations, updated_route_points_distances, removed_count_this_step = \
            update_route_data(
                current_start_index,
                route,
                connections,
                charging_stations_matrix,
                queueing_time
            )

        # Deduct the distance of the segment from the initial EV capacity
        if current_start_index > 0:
            initial_ev_capacity -= route_points_distances[current_start_index - 1]

        # Calculate the number of stations to remove at this step and adjust the population
        current_removed_count = copy.deepcopy(removed_count_this_step) - copy.deepcopy(prev_removed_count)
        adjusted_population_candidates = [x - current_removed_count for x in next_population_candidates]

        # Determine the cluster to which the current starting point belongs
        starting_point_cluster = assigned_points[current_start_index]

        best_charging_stations, _, _ = genetic_algorithm(
            charging_station_points=updated_charging_stations_matrix,
            route_points=updated_route,
            connections=updated_connections,
            initial_population_addition=adjusted_population_candidates,
            population_size=POPULATION_SIZE,
            num_generations=GENERATIONS,
            mutation_rate=MUTATION_RATE,
            queueing_time=updated_queueing_time,
            ev_capacity=EV_CAPACITY,
            initial_ev_capacity=initial_ev_capacity,
            segment_distances=updated_route_points_distances,
            max_stagnation=MAX_STAGNATION,
            cluster_labels=cluster_labels,
            starting_point_cluster=starting_point_cluster
        )

        # Deep copy the best charging stations to avoid modifying the original list
        next_population_candidates = copy.deepcopy(best_charging_stations)

        # Validate and update the final chromosome
        if best_charging_stations and connections[best_charging_stations[0] +
                                                  removed_count_this_step][1] == current_start_index:
            # Remove the first charging station if it matches the starting point
            next_population_candidates.pop(0)
            # Add the first charging station to the final chromosome
            final_chromosome.append(best_charging_stations[0] + removed_count_this_step)

            # Calculate the distance from the charging station to the route
            distances_to_route = calculate_distances_of_cs(charging_stations_matrix, route)
            distance_to_route = distances_to_route[best_charging_stations[0] + removed_count_this_step]
            # Update the remaining EV capacity
            initial_ev_capacity = EV_CAPACITY - distance_to_route

        # Update the number of stations to remove for the next iteration
        prev_removed_count = copy.deepcopy(removed_count_this_step)

        # Store the best route and related data
        best_routes.append((
            updated_route, updated_charging_stations, best_charging_stations, updated_connections,
            updated_queueing_time, updated_route_points_distances, removed_count_this_step
        ))

    return best_routes, final_chromosome


def main():
    # Start the timer
    start_time = time.time()

    # Generate initial data
    route, charging_stations_matrix, queueing_time, route_points_distances, charging_stations = generate_initial_data()

    # Get connections
    connections = get_connections(route, charging_stations)
    # Apply K-means clustering and assign clusters
    cluster_labels, centroids, assigned_points = kmeans_and_assign_clusters(route, charging_stations)

    # Run the genetic algorithm and gather the best routes
    best_routes, final_chromosome = run_genetic_algorithm_for_each_start_point(
        route, connections, cluster_labels, assigned_points, charging_stations_matrix, queueing_time, route_points_distances
    )

    # End the timer and calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    # Calculate fitness score of the final chromosome using the real queueing time (without clustering)
    if final_chromosome:  # Ensure that final_chromosome is not empty
        fitness_score = final_fitness_function(
            final_chromosome, connections, calculate_distances_of_cs(charging_stations_matrix, route),
            queueing_time, EV_CAPACITY, EV_CAPACITY, route_points_distances
        )
        print(f"Best final route: {final_chromosome}")
        print(f"Final Fitness Score: {fitness_score * 100:.6f}")

    # Visualize all routes
    visualize_all_routes(best_routes, cluster_labels, centroids, assigned_points, final_chromosome)


if __name__ == "__main__":
    main()
