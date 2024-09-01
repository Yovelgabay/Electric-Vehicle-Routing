import copy
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from Code.functions import (
    assign_route_points_to_centroids, generate_route_with_checkpoints,
    calculate_route_points_distances, generate_random_charging_stations_and_queueing_time,
    get_intersection_points, closest_point
)
from Code.visualization import visualize_all_routes, visualize_best_route_animation, visualize_clustering
from GA import genetic_algorithm, calculate_distances_of_cs, final_fitness_function
from kmeans import kmeans_clustering
from parameters import *


def generate_initial_data(num_cs_points):
    """Generate initial route, charging stations, and distances."""
    num_route_points = NUM_ROUTE_POINTS
    route = generate_route_with_checkpoints(
        num_route_points, scale=100, turn_amplitude=10, seed=53, checkpoints=CHECK_POINTS)
    charging_stations_matrix, queueing_time = generate_random_charging_stations_and_queueing_time(
        seed=13, num_points=num_cs_points, scale=100, route=route
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

    return (updated_route, updated_connections, updated_charging_stations_matrix, updated_queueing_time,
            updated_charging_stations, updated_route_points_distances, values_to_remove)


def run_genetic_algorithm_for_each_start_point(route, connections, labels, assigned_points, charging_stations_matrix,
                                               queueing_time, route_points_distances, mutation_rate):
    """Run the genetic algorithm for each starting point and gather the best routes."""
    best_routes = []
    final_chromosome = []
    initial_ev_capacity = EV_CAPACITY
    add_to_population = []
    pre_values_to_remove = 0

    for starting_point_index in range(len(route)):
        updated_route, updated_connections, updated_charging_stations_matrix, updated_queueing_time, \
            updated_charging_stations, updated_route_points_distances, values_to_remove = \
            (update_route_data(
                starting_point_index,
                route,
                connections,
                charging_stations_matrix,
                queueing_time))

        if starting_point_index > 0:
            initial_ev_capacity -= route_points_distances[starting_point_index - 1]

        temp_values_to_remove = copy.deepcopy(values_to_remove) - copy.deepcopy(pre_values_to_remove)
        updated_add_to_population = [x - temp_values_to_remove for x in add_to_population]

        starting_point_cluster = assigned_points[starting_point_index]

        best_charging_stations, _, _ = genetic_algorithm(
            updated_charging_stations_matrix, updated_route, updated_connections, updated_add_to_population,
            population_size=POPULATION_SIZE, generations=GENERATIONS, mutation_rate=mutation_rate,
            queueing_time=updated_queueing_time, ev_capacity=EV_CAPACITY, initial_ev_capacity=initial_ev_capacity,
            route_points_distances=updated_route_points_distances, max_stagnation=MAX_STAGNATION, labels=labels,
            starting_point_cluster=starting_point_cluster
        )

        add_to_population = copy.deepcopy(best_charging_stations)

        if best_charging_stations and connections[best_charging_stations[0] +
                                                  values_to_remove][1] == starting_point_index:
            add_to_population.pop(0)
            final_chromosome.append(best_charging_stations[0] + values_to_remove)
            distances_to_route = calculate_distances_of_cs(charging_stations_matrix, route)
            distance_to_route = distances_to_route[best_charging_stations[0] + values_to_remove]
            initial_ev_capacity = EV_CAPACITY - distance_to_route

        pre_values_to_remove = copy.deepcopy(values_to_remove)

        best_routes.append((
            updated_route, updated_charging_stations, best_charging_stations, updated_connections,
            updated_queueing_time, updated_route_points_distances, values_to_remove
        ))

    return best_routes, final_chromosome


def test_mutation_rate():
    # Define the CS points configurations and mutation rates
    cs_points_configurations = [100, 200, 300]
    mutation_rates = np.arange(0, 1.1, 0.1)

    # Store the results for visualization
    results = {cs_points: [] for cs_points in cs_points_configurations}

    for num_cs_points in cs_points_configurations:
        for mutation_rate in mutation_rates:
            print(f"Testing mutation rate: {mutation_rate} with {num_cs_points} CS points")

            # Generate initial data with the specified number of CS points
            route, charging_stations_matrix, queueing_time, route_points_distances, charging_stations = generate_initial_data(
                num_cs_points)

            # Get connections
            connections = get_connections(route, charging_stations)
            # Apply K-means clustering and assign clusters
            labels, centroids, assigned_points = kmeans_and_assign_clusters(route, charging_stations)

            # Run the genetic algorithm with the current mutation rate
            best_routes, final_chromosome = run_genetic_algorithm_for_each_start_point(
                route, connections, labels, assigned_points, charging_stations_matrix, queueing_time,
                route_points_distances, mutation_rate
            )

            # Calculate fitness score of the final chromosome using the real queueing time (without clustering)
            if final_chromosome:
                fitness_score = final_fitness_function(
                    final_chromosome, connections, calculate_distances_of_cs(charging_stations_matrix, route),
                    queueing_time, EV_CAPACITY, EV_CAPACITY, route_points_distances
                )
                results[num_cs_points].append(fitness_score)
                print(
                    f"Mutation Rate: {mutation_rate}, CS Points: {num_cs_points}, Fitness Score: {fitness_score * 100:.6f}")
            else:
                results[num_cs_points].append(0)
                print(f"Mutation Rate: {mutation_rate}, CS Points: {num_cs_points}, Fitness Score: 0")

    # Plotting the results
    plt.figure(figsize=(12, 8))
    for num_cs_points, scores in results.items():
        plt.plot(mutation_rates, scores, marker='o', label=f'{num_cs_points} CS Points')

    plt.xlabel('Mutation Rate')
    plt.ylabel('Best Fitness Score')
    plt.title('Fitness Score vs. Mutation Rate for Different Numbers of Charging Stations')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Run the test on mutation rate
    test_mutation_rate()
