import numpy as np
import random
import copy

from Code.parameters import AVERAGE_QUEUEING_TIME


def calculate_distances_of_cs2(points_with_ids, route):
    """
    Calculate the minimum distance from each charging station to the nearest route point.
    """

    num_points = len(points_with_ids)
    num_route_points = len(route)
    min_distances = np.zeros(num_points)

    # Iterate over each charging station
    for i, (_, (px, py), _) in enumerate(points_with_ids):
        distances = np.zeros(num_route_points)
        # Calculate distance from the charging station to each point on the route
        for j, (qx, qy) in enumerate(route):
            distances[j] = np.hypot(px - qx, py - qy)
        # Find the minimum distance to the nearest route point
        min_distances[i] = distances.min()

    return min_distances


def calculate_distances_of_cs(points_with_ids, route):
    """
    Calculate the minimum distance from each charging station to the nearest route point.
    """

    points_array = np.array([coords for _, coords, _ in points_with_ids])
    route_array = np.array(route)

    # Calculate pairwise distances between charging stations and route points
    distances = np.sqrt(((points_array[:, np.newaxis] - route_array) ** 2).sum(axis=2))
    # Find the minimum distance for each charging station
    min_distances = distances.min(axis=1)

    return min_distances


def check_validity(chromosome, connections, distances, ev_capacity, route_distances, initial_ev_capacity):
    """
    Check if a chromosome's route is valid based on EV capacity constraints.
    """

    total_distance = 0

    # Calculate the distance from the start point to the first charging station
    start_to_first = sum(route_distances[:connections[chromosome[0]][1]]) + distances[chromosome[0]]
    if start_to_first > initial_ev_capacity:
        return False
    total_distance += start_to_first

    # Calculate distances between consecutive charging stations
    for i in range(len(chromosome) - 1):
        start_station = chromosome[i]
        end_station = chromosome[i + 1]

        dist_to_route = distances[start_station]
        route_distance = sum(route_distances[connections[start_station][1]:connections[end_station][1]])
        dist_from_route = distances[end_station]

        segment_distance = dist_to_route + route_distance + dist_from_route

        if segment_distance > ev_capacity:
            return False

        total_distance += segment_distance

    # Calculate the distance from the last charging station to the destination
    last_station_idx = chromosome[-1]
    route_to_end = sum(route_distances[connections[last_station_idx][1]:])
    last_to_end = route_to_end + distances[last_station_idx]
    if last_to_end > ev_capacity:
        return False

    total_distance += last_to_end
    return True


def calculate_exceeded_kilometers(chromosome, connections, distances,
                                  ev_capacity, initial_ev_capacity, route_distances, exceeded_penalty_factor):
    """
    Calculate the total distance exceeding the EV capacity for a given route.
    """

    if check_validity(chromosome, connections, distances, ev_capacity, route_distances, initial_ev_capacity):
        return 0

    total_excess_km = 0

    # Calculate the distance from the start point to the first charging station
    start_to_first = sum(route_distances[:connections[chromosome[0]][1]]) + distances[chromosome[0]]
    if start_to_first > initial_ev_capacity:
        total_excess_km += start_to_first - initial_ev_capacity

    # Calculate distances between consecutive charging stations
    for i in range(len(chromosome) - 1):
        start_station = chromosome[i]
        end_station = chromosome[i + 1]

        dist_to_route = distances[start_station]
        route_distance = sum(route_distances[connections[start_station][1]:connections[end_station][1]])
        dist_from_route = distances[end_station]

        segment_distance = dist_to_route + route_distance + dist_from_route

        if segment_distance > ev_capacity:
            total_excess_km += segment_distance - ev_capacity

    # Calculate the distance from the last charging station to the destination
    last_station_idx = chromosome[-1]
    route_to_end = sum(route_distances[connections[last_station_idx][1]:])
    last_to_end = route_to_end + distances[last_station_idx]
    if last_to_end > ev_capacity:
        total_excess_km += last_to_end - ev_capacity

    return total_excess_km * exceeded_penalty_factor


def fitness_function(chromosome, connections, distances, queueing_time, ev_capacity, initial_ev_capacity,
                     route_distances, cluster_labels, starting_point_cluster):
    """
    Evaluate the fitness of a chromosome based on total distance, queueing time, and penalty for exceeding EV capacity.
    """
    queueing_time_penalty_factor = 2
    # Calculate the total kilometers exceeding the EV capacity
    exceeded_km = calculate_exceeded_kilometers(
        chromosome, connections, distances, ev_capacity,
        initial_ev_capacity, route_distances, exceeded_penalty_factor=200
    )

    # Calculate the total distance traveled by the vehicle
    total_distance = sum(distances[stop] for stop in chromosome)

    # Calculate the queueing time penalty, adjusted based on the cluster of the starting point
    queueing_time_penalty = sum(
        queueing_time[stop] if cluster_labels[stop] == starting_point_cluster
        else AVERAGE_QUEUEING_TIME
        for stop in chromosome
    ) * queueing_time_penalty_factor  # adjust this value as needed

    # Compute the total penalty
    total_penalty = total_distance + queueing_time_penalty + exceeded_km

    # Return the inverse of the total penalty to represent fitness & handle division by zero
    return 1 / total_penalty if total_penalty > 0 else 1e-12


def final_fitness_function(chromosome, connections, distances, queueing_time, ev_capacity,
                           initial_ev_capacity, route_distances):
    """
    Calculate the final fitness score of a chromosome based on total distance, queueing time, and penalty for exceeding EV capacity.
    """

    queueing_time_penalty_factor = 2

    # Calculate the total kilometers exceeding the EV capacity
    exceeded_km = calculate_exceeded_kilometers(
        chromosome, connections, distances, ev_capacity,
        initial_ev_capacity, route_distances, exceeded_penalty_factor=200
    )

    # Calculate the total distance traveled by the vehicle
    total_distance = sum(distances[stop] for stop in chromosome)

    # Calculate the queueing time penalty using the actual queueing times at charging stations
    queueing_time_penalty = sum(queueing_time[stop] for stop in chromosome) * queueing_time_penalty_factor

    # Compute the total penalty
    total_penalty = total_distance + queueing_time_penalty + exceeded_km

    # Return the inverse of the total penalty to represent fitness & handle division by zero
    return 1 / total_penalty if total_penalty > 0 else 1e-12


def tournament_selection(population, fitness_scores, tournament_size):
    """
    Select a parent for crossover using tournament selection.
    """

    indices = list(range(len(population)))

    # Randomly select indices for the tournament
    tournament_indices = random.sample(indices, min(tournament_size, len(indices)))

    # Get the fitness values for the selected indices
    tournament_fitness_scores = [fitness_scores[idx] for idx in tournament_indices]

    # Determine the index of the winner (highest fitness)
    winner_idx = tournament_indices[np.argmax(tournament_fitness_scores)]

    # Select the corresponding parent from the population
    selected_parent = population[winner_idx]

    return selected_parent


def crossover(parent1, parent2):
    """
    Apply crossover operation to two parent chromosomes (routes) to generate two offspring chromosomes.
    """

    set1 = set(parent1)
    set2 = set(parent2)
    common_nodes = list(set1 & set2 - {parent1[0], parent1[-1], parent2[0], parent2[-1]})

    if not common_nodes:
        # Perform crossover without common nodes
        crossover_point = min(len(parent1), len(parent2)) // 2
        offspring1_route = sorted(parent1[:crossover_point] + parent2[crossover_point:])
        offspring2_route = sorted(parent2[:crossover_point] + parent1[crossover_point:])
    else:
        # Perform crossover with common nodes
        crossing_node = random.choice(common_nodes)

        idx1 = parent1.index(crossing_node)
        idx2 = parent2.index(crossing_node)

        child1_part1 = parent1[:idx1]
        child1_part2 = parent2[idx2:]
        offspring1_route = sorted(child1_part1 + child1_part2)

        child2_part1 = parent2[:idx2]
        child2_part2 = parent1[idx1:]
        offspring2_route = sorted(child2_part1 + child2_part2)

    return offspring1_route, offspring2_route


def mutate(route, points_with_ids, mutation_rate):
    """
    Introduce variations into the route by randomly mutating it based on a mutation rate.
    """

    if random.random() > mutation_rate:
        # No mutation occurs, return the original route twice
        return copy.deepcopy(route), copy.deepcopy(route)

    route1 = copy.deepcopy(route)
    route2 = copy.deepcopy(route)

    # If there is only one station in the route, replace it with a different valid station
    if len(route) == 1:
        possible_stations_to_add = [i for i in range(len(points_with_ids)) if i not in route]
        if possible_stations_to_add:
            new_station1 = random.choice(possible_stations_to_add)
            new_station2 = random.choice(possible_stations_to_add)
            route1[0] = new_station1
            route2[0] = new_station2

    # If the route has two stations, add a new station between them
    elif len(route) == 2:
        min_val = route[0]
        max_val = route[1]
        possible_stations = [i for i in range(min_val + 1, max_val) if i not in route]
        if possible_stations:
            new_station1 = random.choice(possible_stations)
            new_station2 = random.choice(possible_stations)
            route1.insert(1, new_station1)
            route2.insert(1, new_station2)

    # If the route has three or more stations, remove one from route1 and add one to route2
    elif len(route) >= 3:
        idx_remove = random.randint(1, len(route1) - 1)
        route1.pop(idx_remove)

        idx_add = random.randint(1, len(route2) - 1)
        min_val = route2[idx_add - 1]
        max_val = route2[idx_add]
        possible_stations = [i for i in range(min_val + 1, max_val) if i not in route2]
        if possible_stations:
            new_station = random.choice(possible_stations)
            route2.insert(idx_add, new_station)
        else:
            idx_remove = random.randint(1, len(route1) - 1)
            route2.pop(idx_remove)

    return route1, route2


def initialize_population(points_with_ids, population_size):
    """
    Generate the initial population for a genetic algorithm. Each individual in the population represents a route
    through the charging stations.
    """

    population = []
    num_points = len(points_with_ids)

    # Add a chromosome with a single randomly chosen charging station (shortest valid path)
    if num_points > 0:
        shortest_path_chromosome = [random.choice(range(num_points))]
        population.append(shortest_path_chromosome)

    # Add a chromosome with a path covering all charging stations (longest path)
    longest_path_chromosome = list(range(num_points))
    population.append(longest_path_chromosome)

    # Add random chromosomes with varying numbers of stops
    for i in range(population_size - 2):
        num_stops = random.randint(1, num_points)  # Number of stops varies randomly
        random_chromosome = sorted(random.sample(range(num_points), num_stops))
        population.append(random_chromosome)

    return population


def evaluate_population(population, connections, distances_CS, queueing_time, ev_capacity, initial_ev_capacity,
                        route_points_distances, cluster_labels, starting_point_cluster):
    """
    Compute the fitness scores for each chromosome (route) in the population. The fitness scores are used to rank
    the chromosomes, and the population is sorted according to these scores.
    """

    fitness_scores = [
        fitness_function(route, connections, distances_CS, queueing_time, ev_capacity, initial_ev_capacity,
                         route_points_distances, cluster_labels, starting_point_cluster)
        for route in population
    ]

    # Pair each route with its fitness score and sort based on fitness score
    evaluated_population = list(zip(population, fitness_scores))
    sorted_population = sorted(evaluated_population, key=lambda x: x[1], reverse=True)

    # Separate sorted routes and fitness scores
    sorted_routes = [route for route, _ in sorted_population]
    sorted_fitness_scores = [fitness for _, fitness in sorted_population]

    return sorted_routes, sorted_fitness_scores


def genetic_algorithm(charging_station_points, route_points, connections, initial_population_addition,
                      population_size, num_generations, mutation_rate, queueing_time, ev_capacity,
                      initial_ev_capacity, segment_distances, max_stagnation, cluster_labels,
                      starting_point_cluster, selection_method):
    """
    Execute the genetic algorithm to find the optimal route based on given parameters.
    """
    # Existing code initialization
    total_route_distance = np.sum(segment_distances)
    if initial_ev_capacity > total_route_distance:
        return [], 1, []

    charging_station_distances = calculate_distances_of_cs(charging_station_points, route_points)
    population = initialize_population(charging_station_points, population_size)

    if len(initial_population_addition) != 0:
        population.append(initial_population_addition)

    evaluated_population, fitness_scores = evaluate_population(
        population, connections, charging_station_distances, queueing_time, ev_capacity,
        initial_ev_capacity, segment_distances, cluster_labels, starting_point_cluster
    )

    best_fitness = fitness_scores[0]
    best_route = evaluated_population[0]
    best_routes_per_generation = [best_route]
    stagnation_counter = 0

    for generation in range(num_generations):
        next_population = []  # Preserve the best route

        # Generate new offspring
        for _ in range(population_size // 4):
            # Use the selected method for parent selection
            if selection_method == 'tournament_4':
                parent1 = tournament_selection(evaluated_population, fitness_scores, tournament_size=4)
                parent2 = tournament_selection(evaluated_population, fitness_scores, tournament_size=4)
            elif selection_method == 'tournament_6':
                parent1 = tournament_selection(evaluated_population, fitness_scores, tournament_size=6)
                parent2 = tournament_selection(evaluated_population, fitness_scores, tournament_size=6)
            elif selection_method == 'tournament_8':
                parent1 = tournament_selection(evaluated_population, fitness_scores, tournament_size=8)
                parent2 = tournament_selection(evaluated_population, fitness_scores, tournament_size=8)
            elif selection_method == 'roulette_wheel':
                parent1 = roulette_wheel_selection(evaluated_population, fitness_scores)
                parent2 = roulette_wheel_selection(evaluated_population, fitness_scores)
            elif selection_method == 'rank_selection':
                parent1 = rank_selection(evaluated_population, fitness_scores)
                parent2 = rank_selection(evaluated_population, fitness_scores)
            else:
                raise ValueError("Invalid selection method specified.")

            child1, child2 = crossover(parent1, parent2)
            mutated_child1a, mutated_child1b = mutate(child1, charging_station_points, mutation_rate)
            mutated_child2a, mutated_child2b = mutate(child2, charging_station_points, mutation_rate)

            next_population.extend([mutated_child1a, mutated_child1b, mutated_child2a, mutated_child2b])

        evaluated_population, fitness_scores = evaluate_population(
            next_population, connections, charging_station_distances, queueing_time,
            ev_capacity, initial_ev_capacity, segment_distances, cluster_labels, starting_point_cluster
        )

        current_best_fitness = fitness_scores[0]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_route = evaluated_population[0]
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        best_routes_per_generation.append(best_route)

        if stagnation_counter >= max_stagnation:
            break

    return best_route, best_fitness, best_routes_per_generation


def roulette_wheel_selection(population, fitness_scores):
    """
    Perform Roulette Wheel Selection on the population.
    Args:
    - population: List of individuals in the population.
    - fitness_scores: List of fitness scores corresponding to each individual.

    Returns:
    - Selected individual from the population.
    """
    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        return np.random.choice(population)  # Random selection if total fitness is zero

    # Calculate the selection probability for each individual
    selection_probabilities = [fitness / total_fitness for fitness in fitness_scores]

    # Choose an individual based on the computed probabilities
    selected_index = np.random.choice(range(len(population)), p=selection_probabilities)
    return population[selected_index]


def rank_selection(population, fitness_scores):
    """
    Perform Rank Selection on the population.
    """

    # Rank individuals by their fitness scores (higher fitness is better)
    sorted_indices = np.argsort(fitness_scores)  # Sort in ascending order
    ranked_population = [population[i] for i in sorted_indices]

    # Assign ranks based on their sorted order
    ranks = np.arange(1, len(population) + 1)  # Rank ranges from 1 to N

    # Calculate selection probability based on rank (higher rank has higher probability)
    total_rank = sum(ranks)
    selection_probabilities = [rank / total_rank for rank in ranks]

    # Choose an individual based on the computed rank probabilities
    selected_index = np.random.choice(range(len(population)), p=selection_probabilities)
    return ranked_population[selected_index]
