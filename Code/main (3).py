import math
import numpy as np
import matplotlib.pyplot as plt
from Code.functions import (
    assign_route_points_to_centroids, generate_route_with_checkpoints,
    calculate_route_points_distances, generate_random_charging_stations_and_queueing_time,
    get_intersection_points, closest_point
)
from Code.visualization import visualize_all_routes, visualize_clustering
from GA import genetic_algorithm, calculate_distances_of_cs, final_fitness_function
from kmeans import kmeans_clustering
from parameters import *


def generate_initial_data(num_points):
    """Generate initial route, charging stations, and distances."""
    num_route_points = NUM_ROUTE_POINTS
    route = generate_route_with_checkpoints(
        num_route_points, scale=100, turn_amplitude=10, seed=53, checkpoints=CHECK_POINTS)
    charging_stations_matrix, queueing_time = generate_random_charging_stations_and_queueing_time(
        seed=13, num_points=num_points, scale=100, route=route
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


def run_genetic_algorithm(route, connections, labels, assigned_points, charging_stations_matrix,
                          queueing_time, route_points_distances, mutation_rate):
    """Run the genetic algorithm for the given mutation rate."""
    starting_point_index = 0
    initial_ev_capacity = EV_CAPACITY

    # Run the genetic algorithm
    best_charging_stations, best_fitness, _ = genetic_algorithm(
        charging_stations_matrix, route, connections,
        population_size=200, generations=50, mutation_rate=mutation_rate,
        queueing_time=queueing_time, ev_capacity=EV_CAPACITY, initial_ev_capacity=initial_ev_capacity,
        route_points_distances=route_points_distances, max_stagnation=10, labels=labels,
        starting_point_cluster=assigned_points[starting_point_index]
    )

    # If a valid solution was found, calculate the fitness score
    if best_charging_stations:
        final_chromosome = [best_charging_stations[0]]
        fitness_score = best_fitness

        return fitness_score * 100  # Multiply to get a percentage representation

    # Return a fitness score of 0 if no valid solution was found
    return 0.0


def test_mutation_rates():
    mutation_rates = np.arange(0, 1.1, 0.1)
    num_cs_points = [100, 200, 300]

    fitness_results = {cs_points: [] for cs_points in num_cs_points}

    for cs_points in num_cs_points:
        route, charging_stations_matrix, queueing_time, route_points_distances, charging_stations = generate_initial_data(
            cs_points)
        connections = get_connections(route, charging_stations)
        labels, centroids, assigned_points = kmeans_and_assign_clusters(route, charging_stations)

        for mutation_rate in mutation_rates:
            fitness_score = run_genetic_algorithm(
                route, connections, labels, assigned_points, charging_stations_matrix, queueing_time,
                route_points_distances, mutation_rate
            )
            fitness_results[cs_points].append(fitness_score)
            print(f"CS Points: {cs_points}, Mutation Rate: {mutation_rate}, Fitness Score: {fitness_score}")

    # Plot results
    plt.figure(figsize=(10, 6))
    for cs_points in num_cs_points:
        plt.plot(mutation_rates, fitness_results[cs_points], marker='o', label=f'{cs_points} CS Points')

    plt.title('Fitness Scores for Different Mutation Rates')
    plt.xlabel('Mutation Rate')
    plt.ylabel('Fitness Score (%)')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Test various mutation rates
    test_mutation_rates()


if __name__ == "__main__":
    main()
