import math
import time
import numpy as np
from Code.functions import (
    assign_route_points_to_centroids, generate_route_with_checkpoints,
    calculate_route_points_distances, generate_random_charging_stations_and_queueing_time,
    get_intersection_points, closest_point
)

from Code.visualization import visualize_all_routes, visualize_best_route_animation ,visualize_clustering
from GA import genetic_algorithm, calculate_distances_of_cs, final_fitness_function
from kmeans import kmeans_clustering
from parameters import *


def generate_initial_data():
    """Generate initial route, charging stations, and distances."""
    num_route_points = NUM_ROUTE_POINTS
    route = generate_route_with_checkpoints(
        num_route_points, scale=100, turn_amplitude=10, seed=53, checkpoints=CHECK_POINTS)
    charging_stations_matrix, queueing_time = generate_random_charging_stations_and_queueing_time(
        seed=13, num_points=NUM_POINTS, scale=100, route=route
    )
    route_points_distances = calculate_route_points_distances(route)
    charging_stations = np.array([charging_station[1] for charging_station in charging_stations_matrix])
    return route, charging_stations_matrix, queueing_time, route_points_distances, charging_stations


def get_connections(route, charging_stations):
    """Find intersection points and connections between route and charging stations."""
    intersections = get_intersection_points(route, charging_stations)
    connections = [(idx, closest_point(route, pt)) for idx, pt, _ in intersections]
    return connections


def kmeans_and_assign_clusters(route, charging_stations):
    """Apply K-means clustering and assign route points to clusters."""
    labels, centroids, num_clusters = kmeans_clustering(charging_stations, math.ceil(len(route) / 3))
    assigned_points = assign_route_points_to_centroids(centroids, route)
    visualize_clustering(num_clusters, charging_stations, labels, centroids)
    return labels, centroids, assigned_points


def update_route_data(starting_point_index, route, connections, charging_stations_matrix, queueing_time):
    """Update route, connections, charging stations, and distances based on the starting point."""
    updated_route = route[starting_point_index:]
    updated_connections = [(x, y - starting_point_index) for (x, y) in connections if y >= starting_point_index]
    values_to_remove = len(connections) - len(updated_connections)

    updated_charging_stations_matrix = charging_stations_matrix[values_to_remove:]
    updated_queueing_time = queueing_time[values_to_remove:]
    updated_charging_stations = np.array([point[1] for point in updated_charging_stations_matrix])
    updated_route_points_distances = calculate_route_points_distances(updated_route)

    return updated_route, updated_connections, updated_charging_stations_matrix, updated_queueing_time, updated_charging_stations, updated_route_points_distances, values_to_remove


def run_genetic_algorithm_for_each_start_point(route, connections, labels, assigned_points, charging_stations_matrix,
                                               queueing_time, route_points_distances):
    """Run the genetic algorithm for each starting point and gather the best routes."""
    best_routes = []
    final_chromosome = []
    initial_ev_capacity = EV_CAPACITY

    for starting_point_index in range(len(route)):
        print(f"Starting Point Index: {starting_point_index}")

        # Update route and related data
        updated_route, updated_connections, updated_charging_stations_matrix, updated_queueing_time, \
            updated_charging_stations, updated_route_points_distances, values_to_remove = \
            (update_route_data(
                starting_point_index,
                route,
                connections,
                charging_stations_matrix,
                queueing_time))

        # Deduct segment distance from initial_ev_capacity
        if starting_point_index > 0:
            initial_ev_capacity -= route_points_distances[starting_point_index - 1]
            print("initial_ev_capacity", initial_ev_capacity)

        # Determine the starting point cluster
        starting_point_cluster = assigned_points[starting_point_index]

        # Run the genetic algorithm
        best_charging_stations, _, _ = genetic_algorithm(
            updated_charging_stations_matrix, updated_route, updated_connections,
            population_size=POPULATION_SIZE, generations=GENERATIONS, mutation_rate=MUTATION_RATE,
            queueing_time=updated_queueing_time, ev_capacity=EV_CAPACITY, initial_ev_capacity=initial_ev_capacity,
            route_points_distances=updated_route_points_distances, max_stagnation=MAX_STAGNATION, labels=labels,
            starting_point_cluster=starting_point_cluster
        )

        # Validate and update final chromosome
        if best_charging_stations and connections[best_charging_stations[0] +
                                                  values_to_remove][1] == starting_point_index:
            final_chromosome.append(best_charging_stations[0] + values_to_remove)
            distances_to_route = calculate_distances_of_cs(charging_stations_matrix, route)
            distance_to_route = distances_to_route[best_charging_stations[0] + values_to_remove]
            initial_ev_capacity = EV_CAPACITY - distance_to_route

        # Store the best route
        best_routes.append((
            updated_route, updated_charging_stations, best_charging_stations, updated_connections,
            updated_queueing_time, updated_route_points_distances, values_to_remove
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
    labels, centroids, assigned_points = kmeans_and_assign_clusters(route, charging_stations)
    # _, _, generations_data = genetic_algorithm(charging_stations_matrix, NUM_ROUTE_POINTS, connections,
    # POPULATION_SIZE, GENERATIONS, MUTATION_RATE, queueing_time, EV_CAPACITY, EV_CAPACITY, route_points_distances,
    # MAX_STAGNATION, labels, 0) visualize_best_route_animation(route, charging_stations, generations_data,
    # connections, route_points_distances, queueing_time, interval=1000)

    # Run the genetic algorithm and gather the best routes
    best_routes, final_chromosome = run_genetic_algorithm_for_each_start_point(
        route, connections, labels, assigned_points, charging_stations_matrix, queueing_time, route_points_distances
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
    visualize_all_routes(best_routes, labels, centroids, assigned_points)


if __name__ == "__main__":
    main()
