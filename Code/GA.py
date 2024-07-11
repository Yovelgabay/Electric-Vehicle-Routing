import numpy as np
import random
import copy

from Code.parameters import AVERAGE_WAITING_TIME


def point_to_point_distance(px, py, qx, qy):
    return np.hypot(px - qx, py - qy)


def calculate_distances_of_cs(points_with_ids, route):
    num_points = len(points_with_ids)
    num_route_points = len(route)
    min_distances = np.zeros(num_points)

    for i, (_, (px, py), _) in enumerate(points_with_ids):
        distances = np.zeros(num_route_points)
        for j, (qx, qy) in enumerate(route):
            distances[j] = point_to_point_distance(px, py, qx, qy)
        min_distances[i] = distances.min()

    return min_distances


def check_validity(chromosome, connections, distances, ev_capacity, route_distances):
    total_distance = 0

    # Calculate the distance from the start point to the first charging station
    start_to_first = sum(route_distances[:connections[chromosome[0]][1]])
    start_to_first += distances[chromosome[0]]
    if start_to_first > ev_capacity:
        return False
    total_distance += start_to_first

    # Calculate distances between consecutive charging stations and ensure they do not exceed EV capacity
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


def calculate_exceeded_kilometers(route, connections, distances, ev_capacity, route_distances):
    total_excess_km = 0

    # Calculate the distance from the start point to the first charging station
    start_to_first = sum(route_distances[:connections[route[0]][1]]) + distances[route[0]]
    if start_to_first > ev_capacity:
        total_excess_km += start_to_first - ev_capacity

    # Calculate distances between consecutive charging stations
    for i in range(len(route) - 1):
        start_station = route[i]
        end_station = route[i + 1]

        dist_to_route = distances[start_station]
        route_distance = sum(route_distances[connections[start_station][1]:connections[end_station][1]])
        dist_from_route = distances[end_station]

        segment_distance = dist_to_route + route_distance + dist_from_route

        if segment_distance > ev_capacity:
            total_excess_km += segment_distance - ev_capacity

    # Calculate the distance from the last charging station to the destination
    last_station_idx = route[-1]
    route_to_end = sum(route_distances[connections[last_station_idx][1]:])
    last_to_end = route_to_end + distances[last_station_idx]
    if last_to_end > ev_capacity:
        total_excess_km += last_to_end - ev_capacity

    return total_excess_km


def fitness_function(chromosome, connections, distances, penalties, ev_capacity, route_distances, labels,
                     starting_point_cluster):
    exceeded_km = 0
    if not check_validity(chromosome, connections, distances, ev_capacity, route_distances):
        exceeded_km = 100 * calculate_exceeded_kilometers(chromosome, connections, distances, ev_capacity,
                                                         route_distances)

    total_distance = sum(distances[stop] for stop in chromosome)
    total_penalty = sum(
        penalties[stop] if labels[stop] == starting_point_cluster else AVERAGE_WAITING_TIME
        for stop in chromosome
    )
    stop_penalty = len(chromosome) * 10
    return 1 / (total_distance + total_penalty + stop_penalty + exceeded_km) \
        if total_distance + total_penalty + stop_penalty + exceeded_km > 0 else 0.000000000001


def tournament_selection(population, fitnesses, tournament_size):
    indices = list(range(len(population)))

    # Randomly select indices for the tournament
    tournament_indices = random.sample(indices, min(tournament_size, len(indices)))

    # Get the fitness values for the selected indices
    tournament_fitnesses = [fitnesses[idx] for idx in tournament_indices]

    # Determine the index of the winner (highest fitness)
    winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]

    # Select the corresponding parent from the population
    selected_parent = population[winner_idx]

    return selected_parent


def crossover(parent1, parent2):
    size1, size2 = len(parent1), len(parent2)
    set1 = set(parent1)
    set2 = set(parent2)
    common_nodes = list(
        set1 & set2 - set([parent1[0], parent1[-1], parent2[0], parent2[-1]]))

    if not common_nodes:
        return parent1, parent2

    crossing_node = random.choice(common_nodes)

    idx1 = parent1.index(crossing_node)
    idx2 = parent2.index(crossing_node)

    child1_part1 = parent1[:idx1]
    child1_part2 = parent2[idx2:]
    child1 = child1_part1 + child1_part2

    child2_part1 = parent2[:idx2]
    child2_part2 = parent1[idx1:]
    child2 = child2_part1 + child2_part2

    return child1, child2

"""
def mutate(route, mutation_rate, points_with_ids):
    if random.random() < mutation_rate:
        idx = random.randint(0, len(route) - 1)
        if idx == 0:
            min_val = 0
        else:
            min_val = route[idx - 1]
        if idx == len(route) - 1:
            max_val = len(points_with_ids) - 1
        else:
            max_val = route[idx + 1]
        possible_stations = [i for i in range(min_val + 1, max_val) if i != route[idx]]
        if possible_stations:
            new_station = random.choice(possible_stations)
            route[idx] = new_station

    return route
 """


def mutate(route, points_with_ids):
    """
    Mutates a given chromosome by either replacing, adding, or swapping charging stations.

    Parameters:
    - chromosome (dict): Chromosome with a route.
    - points_with_ids (list): List of tuples (index, coordinates, closest route segment index) for each charging station.

    Returns:
    - mutated_chromosome1, mutated_chromosome2 (tuple): Two mutated chromosomes.
    """
    route1 = copy.deepcopy(route)
    route2 = copy.deepcopy(route)

    if len(route) == 1:
        possible_stations_to_add = [i for i in range(len(points_with_ids)) if i not in route]
        if possible_stations_to_add:
            new_station1 = random.choice(possible_stations_to_add)
            new_station2 = random.choice(possible_stations_to_add)
            route1[0] = new_station1
            route2[0] = new_station2

    # If route has two stations, create two new routes by adding a station between the two
    elif len(route) == 2:
        min_val = route[0]
        max_val = route[1]
        possible_stations = [i for i in range(min_val + 1, max_val) if i not in route]
        if possible_stations:
            new_station1 = random.choice(possible_stations)
            new_station2 = random.choice(possible_stations)
            route1.insert(1, new_station1)
            route2.insert(1, new_station2)


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
    """Create a population of chromosomes"""
    population = []
    num_points = len(points_with_ids)

    # Add a chromosome with the shortest valid path (at least one intermediate stop)
    if num_points > 0:
        shortest_path_chromosome = [random.choice(range(num_points))]
        population.append(shortest_path_chromosome)
        # print(f"Shortest path chromosome: {shortest_path_chromosome}")

    # Add a chromosome with the longest path (going through all charging stations)
    longest_path_chromosome = list(range(num_points))
    population.append(longest_path_chromosome)
    # print(f"Longest path chromosome: {longest_path_chromosome}")

    # Add chromosomes with random paths between origin and destination until we reach the total number of individuals
    # in the population
    for i in range(population_size - 2):
        num_stops = random.randint(1, num_points)  # Vary the number of stops
        stops = sorted(random.sample(range(num_points), num_stops))
        random_chromosome = stops
        population.append(random_chromosome)
        # print(f"Random chromosome {i + 2}: {random_chromosome}")
    return population


'''
def initialize_population(points_with_ids, population_size):
    population = []
    num_points = len(points_with_ids)
    for i in range(population_size):
        num_stops = random.randint(1, num_points)  # Vary the number of stops
        stops = sorted(random.sample(range(num_points), num_stops))
        population.append(stops)
        print(f"Chromosome {i}: {stops}")
    return population
'''


def evaluate_population(population, connections, distances_CS, penalties, ev_capacity, distances_between_points, labels,
                        starting_point_cluster):
    fitnesses = [
        fitness_function(route, connections, distances_CS, penalties, ev_capacity, distances_between_points, labels,
                         starting_point_cluster)
        for route in population
    ]

    evaluated_population = list(zip(population, fitnesses))
    sorted_population = sorted(evaluated_population, key=lambda x: x[1], reverse=True)
    sorted_routes = [route for route, _ in sorted_population]
    sorted_fitnesses = [fitness for _, fitness in sorted_population]

    return sorted_routes, sorted_fitnesses


def genetic_algorithm(points_with_ids, route_points, connections, population_size, generations, mutation_rate,
                      penalties,
                      ev_capacity, distances_between_points, max_stagnation, labels, starting_point_cluster):
    distances_CS = calculate_distances_of_cs(points_with_ids, route_points)

    population = initialize_population(points_with_ids, population_size)
    evaluated_population, fitnesses = evaluate_population(population, connections, distances_CS, penalties,
                                                          ev_capacity, distances_between_points, labels,
                                                          starting_point_cluster)

    best_fitness = fitnesses[0]
    best_route = evaluated_population[0]

    stagnation_counter = 0
    for generation in range(generations):
        next_population = []
        for _ in range(population_size // 4):
            parent1 = tournament_selection(evaluated_population, fitnesses, tournament_size=5)
            parent2 = tournament_selection(evaluated_population, fitnesses, tournament_size=5)

            child1, child2 = crossover(parent1, parent2)
            mutated_child1a, mutated_child1b = mutate(child1, points_with_ids)
            mutated_child2a, mutated_child2b = mutate(child2, points_with_ids)

            next_population.extend([mutated_child1a, mutated_child1b, mutated_child2a, mutated_child2b])

        evaluated_population, fitnesses = evaluate_population(next_population, connections, distances_CS, penalties,
                                                              ev_capacity, distances_between_points, labels,
                                                              starting_point_cluster)

        current_best_fitness = fitnesses[0]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_route = evaluated_population[0]
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= max_stagnation:
            break

        print(f"Generation {generation + 1}: Best fitness = {best_fitness}")

    return best_route, best_fitness
