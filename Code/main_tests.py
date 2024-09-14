import copy
import math
import time
import os
from matplotlib import pyplot as plt

from Code.functions import (
    assign_route_points_to_centroids, generate_route_with_checkpoints,
    calculate_route_points_distances, generate_random_charging_stations_and_queueing_time,
    get_intersection_points, closest_point
)

from Code.visualization import visualize_all_routes, visualize_best_route_animation, visualize_clustering
from ga_tests import genetic_algorithm, calculate_distances_of_cs, final_fitness_function
from kmeans import kmeans_clustering
from parameters import *

os.environ['LOKY_MAX_CPU_COUNT'] = '4'


def generate_initial_data():
    """
    Generate initial data for the routing and charging station setup.
    """
    num_route_points = NUM_ROUTE_POINTS
    route = generate_route_with_checkpoints(
        num_route_points, scale=100, turn_amplitude=10, seed=53, checkpoints=CHECK_POINTS
    )
    charging_stations_matrix, queueing_time = generate_random_charging_stations_and_queueing_time(
        seed=13, num_points=NUM_POINTS, scale=100, route=route
    )
    route_points_distances = calculate_route_points_distances(route)
    charging_stations = np.array([charging_station[1] for charging_station in charging_stations_matrix])

    return route, charging_stations_matrix, queueing_time, route_points_distances, charging_stations


def get_connections(route, charging_stations):
    intersections = get_intersection_points(route, charging_stations)
    connections = [(idx, closest_point(route, pt)) for idx, pt, _ in intersections]
    return connections


def kmeans_and_assign_clusters(route, charging_stations):
    cluster_labels, centroids, num_clusters = kmeans_clustering(charging_stations, math.ceil(len(route) / 3))
    assigned_points = assign_route_points_to_centroids(centroids, route)
    return cluster_labels, centroids, assigned_points


def update_route_data(starting_point_index, route, connections, charging_stations_matrix, queueing_time):
    updated_route = route[starting_point_index:]
    updated_connections = [(x, y - starting_point_index) for (x, y) in connections if y >= starting_point_index]
    values_to_remove = len(connections) - len(updated_connections)
    updated_charging_stations_matrix = charging_stations_matrix[values_to_remove:]
    updated_queueing_time = queueing_time[values_to_remove:]
    updated_charging_stations = np.array([point[1] for point in updated_charging_stations_matrix])
    updated_route_points_distances = calculate_route_points_distances(updated_route)

    return updated_route, updated_connections, updated_charging_stations_matrix, updated_queueing_time, updated_charging_stations, updated_route_points_distances


def run_genetic_algorithm_for_each_start_point(route, connections, cluster_labels, assigned_points, charging_stations_matrix, queueing_time, route_points_distances, pop_size, add_to_initial):
    best_routes = []
    final_chromosome = []
    initial_ev_capacity = EV_CAPACITY
    next_population_candidates = []
    prev_removed_count = 0

    for current_start_index in range(len(route)):
        updated_route, updated_connections, updated_charging_stations_matrix, updated_queueing_time, updated_charging_stations, updated_route_points_distances, removed_count_this_step = update_route_data(
            current_start_index, route, connections, charging_stations_matrix, queueing_time)

        if current_start_index > 0:
            initial_ev_capacity -= route_points_distances[current_start_index - 1]

        current_removed_count = copy.deepcopy(removed_count_this_step) - copy.deepcopy(prev_removed_count)
        adjusted_population_candidates = [x - current_removed_count for x in next_population_candidates]
        starting_point_cluster = assigned_points[current_start_index]

        best_charging_stations, _, _ = genetic_algorithm(
            charging_station_points=charging_stations_matrix,
            route_points=route,
            connections=connections,
            initial_population_addition=adjusted_population_candidates,
            population_size=pop_size,
            num_generations=GENERATIONS,
            mutation_rate=1,
            queueing_time=queueing_time,
            ev_capacity=EV_CAPACITY,
            initial_ev_capacity=100,
            segment_distances=route_points_distances,
            max_stagnation=MAX_STAGNATION,
            cluster_labels=cluster_labels,
            starting_point_cluster=starting_point_cluster,
            selection_method='tournament_4',
            mutation_method='variant',
            crossover_method='variant',
            include_best_route=False,
            add_to_initial=add_to_initial
        )

        next_population_candidates = copy.deepcopy(best_charging_stations)
        index_to_check = best_charging_stations[0] + removed_count_this_step

        if (0 <= index_to_check < len(connections)) and connections[index_to_check][1] == current_start_index:
            next_population_candidates.pop(0)
            final_chromosome.append(index_to_check)
            distances_to_route = calculate_distances_of_cs(charging_stations_matrix, route)
            distance_to_route = distances_to_route[index_to_check]
            initial_ev_capacity = EV_CAPACITY - distance_to_route

        prev_removed_count = copy.deepcopy(removed_count_this_step)
        best_routes.append((
            updated_route, updated_charging_stations, best_charging_stations, updated_connections,
            updated_queueing_time, updated_route_points_distances, removed_count_this_step
        ))

    return best_routes, final_chromosome


# Experiment 1: Mutation Rate Experiment
def run_experiment_with_mutation_rates():
    start_time = time.time()
    route, charging_stations_matrix, queueing_time, route_points_distances, charging_stations = generate_initial_data()
    connections = get_connections(route, charging_stations)
    cluster_labels, centroids, assigned_points = kmeans_and_assign_clusters(route, charging_stations)

    mutation_methods = ['variant', 'random_resetting']
    mutation_rates = [i * 0.1 for i in range(11)]
    population_sizes = [1000]

    results = {size: {rate: {} for rate in mutation_rates} for size in population_sizes}

    for size in population_sizes:
        for rate in mutation_rates:
            for method in mutation_methods:
                best_charging_stations, final_fitness, _ = genetic_algorithm(
                    charging_station_points=charging_stations_matrix,
                    route_points=route,
                    connections=connections,
                    initial_population_addition=[],
                    population_size=size,
                    num_generations=GENERATIONS,
                    mutation_rate=rate,
                    queueing_time=queueing_time,
                    ev_capacity=EV_CAPACITY,
                    initial_ev_capacity=100,
                    segment_distances=route_points_distances,
                    max_stagnation=MAX_STAGNATION,
                    cluster_labels=cluster_labels,
                    starting_point_cluster=assigned_points[0],
                    selection_method='tournament_4',
                    mutation_method=method,
                    crossover_method='variant',
                    include_best_route=False,
                    add_to_initial=True
                )
                results[size][rate][method] = final_fitness

    visualize_mutation_rates(results, mutation_rates, mutation_methods)
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")


# Experiment 2: Selection Methods Experiment
def run_experiment_with_selection_methods():
    start_time = time.time()
    route, charging_stations_matrix, queueing_time, route_points_distances, charging_stations = generate_initial_data()
    connections = get_connections(route, charging_stations)
    cluster_labels, centroids, assigned_points = kmeans_and_assign_clusters(route, charging_stations)

    selection_methods = ['tournament_4', 'tournament_6', 'tournament_8', 'roulette_wheel', 'rank_selection']
    population_sizes = [200, 500, 1000, 2000]
    results = {size: {} for size in population_sizes}

    for size in population_sizes:
        for method in selection_methods:
            best_charging_stations, final_fitness, _ = genetic_algorithm(
                charging_station_points=charging_stations_matrix,
                route_points=route,
                connections=connections,
                initial_population_addition=[],
                population_size=size,
                num_generations=GENERATIONS,
                mutation_rate=1,
                queueing_time=queueing_time,
                ev_capacity=EV_CAPACITY,
                initial_ev_capacity=100,
                segment_distances=route_points_distances,
                max_stagnation=MAX_STAGNATION,
                cluster_labels=cluster_labels,
                starting_point_cluster=assigned_points[0],
                selection_method=method,
                mutation_method='variant',
                crossover_method='variant'
            )
            results[size][method] = final_fitness

    visualize_selections_methods(results, population_sizes, selection_methods)
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")


# Experiment 3: Crossover Methods Experiment
def run_experiment_with_crossover_methods():
    start_time = time.time()
    route, charging_stations_matrix, queueing_time, route_points_distances, charging_stations = generate_initial_data()
    connections = get_connections(route, charging_stations)
    cluster_labels, centroids, assigned_points = kmeans_and_assign_clusters(route, charging_stations)

    crossover_methods = ['variant', 'double_point', 'uniform']
    population_sizes = [200, 500, 1000, 2000]
    results = {size: {} for size in population_sizes}

    for size in population_sizes:
        for method in crossover_methods:
            best_charging_stations, final_fitness, _ = genetic_algorithm(
                charging_station_points=charging_stations_matrix,
                route_points=route,
                connections=connections,
                initial_population_addition=[],
                population_size=size,
                num_generations=GENERATIONS,
                mutation_rate=1,
                queueing_time=queueing_time,
                ev_capacity=EV_CAPACITY,
                initial_ev_capacity=100,
                segment_distances=route_points_distances,
                max_stagnation=MAX_STAGNATION,
                cluster_labels=cluster_labels,
                starting_point_cluster=assigned_points[0],
                selection_method='tournament_4',
                mutation_method='variant',
                crossover_method=method
            )
            results[size][method] = final_fitness

    visualize_crossover_methods(results)
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")


# Experiment 4: Best Route Experiments
def run_experiment_with_best_route():
    start_time = time.time()
    route, charging_stations_matrix, queueing_time, route_points_distances, charging_stations = generate_initial_data()
    connections = get_connections(route, charging_stations)
    cluster_labels, centroids, assigned_points = kmeans_and_assign_clusters(route, charging_stations)

    population_sizes = [200, 500, 1000, 2000]
    include_best_options = [True, False]
    results = {}

    for size in population_sizes:
        for include_best in include_best_options:
            best_charging_stations, final_fitness, _ = genetic_algorithm(
                charging_station_points=charging_stations_matrix,
                route_points=route,
                connections=connections,
                initial_population_addition=[],
                population_size=size,
                num_generations=GENERATIONS,
                mutation_rate=1,
                queueing_time=queueing_time,
                ev_capacity=EV_CAPACITY,
                initial_ev_capacity=100,
                segment_distances=route_points_distances,
                max_stagnation=MAX_STAGNATION,
                cluster_labels=cluster_labels,
                starting_point_cluster=assigned_points[0],
                selection_method='tournament_4',
                mutation_method='variant',
                crossover_method='variant',
                include_best_route=include_best
            )
            results[(size, include_best)] = final_fitness

    visualize_best_route_experiment_results(results)
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")


# Visualization functions
def visualize_mutation_rates(results, mutation_rates, mutation_methods):
    plt.figure(figsize=(12, 6))
    for size in results.keys():
        for method in mutation_methods:
            fitness_values = [results[size][rate][method] for rate in mutation_rates]
            plt.plot(mutation_rates, fitness_values, label=f"Method: {method}")
    plt.xlabel("Mutation Rate")
    plt.ylabel("Fitness Score")
    plt.title("Impact of Mutation Rate on Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_selections_methods(results, population_sizes, selection_methods):
    plt.figure(figsize=(12, 8))
    x = range(len(selection_methods))
    width = 0.2
    for i, size in enumerate(population_sizes):
        fitness_scores = [results[size][method] for method in selection_methods]
        plt.bar([pos + i * width for pos in x], fitness_scores, width=width, label=f'Population Size: {size}')
        for j, score in enumerate(fitness_scores):
            plt.text(j + i * width, score + 0.01, round(score, 2), ha='center', va='bottom', fontsize=10)
    plt.xlabel('Selection Methods')
    plt.ylabel('Fitness Scores')
    plt.title('Fitness Scores for Different Selection Methods Across Population Sizes')
    plt.xticks([pos + width for pos in x], selection_methods)
    plt.legend(title='Population Size')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def visualize_crossover_methods(results):
    crossover_methods = list(next(iter(results.values())).keys())
    population_sizes = sorted(results.keys())
    bar_width = 0.2
    indices = np.arange(len(crossover_methods))

    plt.figure(figsize=(12, 6))
    for i, size in enumerate(population_sizes):
        fitness_scores = [results[size][method] for method in crossover_methods]
        plt.bar(indices + i * bar_width, fitness_scores, width=bar_width, label=f'Population Size: {size}')
    plt.title('Fitness Scores by Population Size and Crossover Method')
    plt.xlabel('Crossover Method')
    plt.ylabel('Fitness Score')
    plt.xticks(indices + bar_width * (len(population_sizes) - 1) / 2, crossover_methods)
    plt.legend(title='Population Size')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def visualize_best_route_experiment_results(results):
    population_sizes = sorted(set(key[0] for key in results.keys()))
    include_best_options = sorted(set(key[1] for key in results.keys()))

    include_best_fitness = {pop_size: [] for pop_size in population_sizes}
    exclude_best_fitness = {pop_size: [] for pop_size in population_sizes}

    for (pop_size, include_best), fitness_score in results.items():
        if include_best:
            include_best_fitness[pop_size].append(fitness_score)
        else:
            exclude_best_fitness[pop_size].append(fitness_score)

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(population_sizes))
    width = 0.35
    include_best_means = [np.mean(include_best_fitness[pop_size]) for pop_size in population_sizes]
    exclude_best_means = [np.mean(exclude_best_fitness[pop_size]) for pop_size in population_sizes]

    ax.bar(x - width / 2, include_best_means, width, label='Include Best Route', color='blue')
    ax.bar(x + width / 2, exclude_best_means, width, label='Exclude Best Route', color='orange')

    ax.set_xlabel('Population Size')
    ax.set_ylabel('Fitness Score')
    ax.set_title('Genetic Algorithm Performance with Different Population Sizes')
    ax.set_xticks(x)
    ax.set_xticklabels(population_sizes)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment_with_mutation_rates()
    run_experiment_with_selection_methods()
    run_experiment_with_crossover_methods()
    run_experiment_with_best_route()
