import numpy as np
import random
import copy

from scipy.spatial.distance import cdist


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """
    Calculates the shortest distance from a point (px, py) to a line segment defined by (x1, y1) and (x2, y2).

    Parameters:
    - px (float): X-coordinate of the point.
    - py (float): Y-coordinate of the point.
    - x1 (float): X-coordinate of the start of the line segment.
    - y1 (float): Y-coordinate of the start of the line segment.
    - x2 (float): X-coordinate of the end of the line segment.
    - y2 (float): Y-coordinate of the end of the line segment.

    Returns:
    - distance (float): The shortest distance from the point to the segment.
    """
    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return np.hypot(px - x1, py - y1)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    closest_x, closest_y = x1 + t * dx, y1 + t * dy
    return np.hypot(px - closest_x, py - closest_y)


def calculate_distances_of_CS(points_with_ids, route):
    """
    Calculates the minimum distance from each charging station to the nearest segment of the route.

    Parameters:
    - points_with_ids (list): List of tuples (index, coordinates, closest route segment index) for each charging station.
    - route (np.ndarray): Array of route waypoints.

    Returns:
    - min_distances (np.ndarray): Array of the shortest distances from each charging station to the route.
    """
    distances = np.zeros((len(points_with_ids), len(route) - 1))
    for i, (_, (px, py), _) in enumerate(points_with_ids):
        for j in range(len(route) - 1):
            x1, y1 = route[j]
            x2, y2 = route[j + 1]
            distances[i, j] = point_to_segment_distance(px, py, x1, y1, x2, y2)
    min_distances = distances.min(axis=1)
    return min_distances


def check_validity(chromosome, connections, distances, ev_capacity, route_distances):
    """
    Checks if a given route (chromosome) is valid by ensuring the total distance between consecutive charging stations
    and the start/end of the route does not exceed the EV's capacity.

    Parameters:
    - chromosome (list): List of indices representing the route of the EV through charging stations.
    - connections (list): List of tuples defining connections between points and route segments.
    - distances (np.ndarray): Array of distances from charging stations to the route.
    - ev_capacity (float): Maximum distance the EV can travel on a single charge.
    - route_distances (list): List of distances between consecutive waypoints on the route.

    Returns:
    - bool: True if the route is valid, False otherwise.
    """
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


def fitness_function(chromosome, connections, distances, penalties, ev_capacity, route_distances):
    """
    Evaluates the fitness of a given route (chromosome) by considering the total distance, penalties, and the number
    of stops. Higher fitness values indicate a more optimal route.

    Parameters:
    - chromosome (list): List of indices representing the route of the EV through charging stations.
    - connections (list): List of tuples defining connections between points and route segments.
    - distances (np.ndarray): Array of distances from charging stations to the route.
    - penalties (np.ndarray): Array of penalty values assigned to each station.
    - ev_capacity (float): Maximum distance the EV can travel on a single charge.
    - route_distances (list): List of distances between consecutive waypoints on the route.

    Returns:
    - fitness (float): Fitness value of the route, higher values are better.
    """
    if not check_validity(chromosome, connections, distances, ev_capacity, route_distances):
        return 0.0000000001  # Return a small fitness if the chromosome is not valid

    total_distance = sum(distances[stop] for stop in chromosome)
    total_penalty = sum(penalties[stop] for stop in chromosome)
    stop_penalty = len(chromosome)
    return 1 / (total_distance + total_penalty + stop_penalty) if total_distance + total_penalty + stop_penalty > 0 else 0.000000000001


def tournament_selection(population, fitnesses, tournament_size=2):
    """
    Selects parents for crossover based on a tournament selection method.

    Parameters:
    - population (list): List of routes (chromosomes) in the current generation.
    - fitnesses (list): List of fitness values corresponding to the routes in the population.
    - tournament_size (int): Size of the tournament group for selection.

    Returns:
    - selected_parents (list): List of selected parent routes for crossover.
    """
    selected_parents = []
    indices = list(range(len(population)))

    while indices:
        tournament_indices = random.sample(indices, min(tournament_size, len(indices)))
        tournament_fitnesses = [fitnesses[idx] for idx in tournament_indices]

        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        selected_parents.append(population[winner_idx])

        indices.remove(winner_idx)

    return selected_parents


def crossover(parent1, parent2):
    """
    Performs crossover between two parent routes at a common node, creating two offspring routes by exchanging segments
    at the crossing node.

    Parameters:
    - parent1 (list): List of indices representing the first parent route.
    - parent2 (list): List of indices representing the second parent route.

    Returns:
    - child1 (list): First offspring route resulting from the crossover operation.
    - child2 (list): Second offspring route resulting from the crossover operation.
    """
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


def mutate(route, mutation_rate, points_with_ids):
    """
    Mutates a given route by randomly changing one of the charging stations in the route with another valid station.

    Parameters:
    - route (list): List of indices representing the route of the EV through charging stations.
    - mutation_rate (float): Probability of mutation for the route.
    - points_with_ids (list): List of tuples (index, coordinates, closest route segment index) for each charging station.

    Returns:
    - route (list): Mutated route.
    """
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

def initialize_population(points_with_ids, population_size):
    """
    Initializes a population of routes (chromosomes) for the genetic algorithm.

    Parameters:
    - points_with_ids (list): List of tuples containing indices, coordinates, and closest route point indices.
    - population_size (int): Number of chromosomes (routes) to generate.

    Returns:
    - list: List of lists, where each inner list represents a chromosome (route).
      Each chromosome is a sorted list of indices representing charging stations along the route.
    """
    population = []
    num_points = len(points_with_ids)
    for i in range(population_size):
        num_stops = random.randint(1, num_points)  # Vary the number of stops
        stops = sorted(random.sample(range(num_points), num_stops))
        population.append(stops)
        print(f"Chromosome {i}: {stops}")
    return population


def genetic_algorithm(points_with_ids, route, connections, population_size, generations, mutation_rate, penalties,
                      ev_capacity, distances_between_points):
    """
    Executes a genetic algorithm to optimize the EV route selection based on charging station locations.

    Parameters:
    - points_with_ids (list): List of tuples containing indices, coordinates, and closest route point indices.
    - route (ndarray): Array of coordinates representing the route waypoints.
    - connections (list): List of tuples indicating connections between charging stations and route waypoints.
    - population_size (int): Number of chromosomes (routes) in each generation.
    - generations (int): Number of generations (iterations) to run the genetic algorithm.
    - mutation_rate (float): Probability of mutation for each chromosome during reproduction.
    - penalties (ndarray): Array of penalties associated with each charging station.
    - ev_capacity (float): Maximum distance the EV can travel on a single charge.
    - distances_between_points (ndarray): Distances between consecutive route waypoints.

    Returns:
    - list: Best route found by the genetic algorithm, represented as a list of indices of charging stations.
    """
    print("Initializing population...")
    population = initialize_population(points_with_ids, population_size)
    distances_CS = calculate_distances_of_CS(points_with_ids, route)
    best_route = None
    best_fitness = float('-inf')

    for generation in range(generations):
        fitnesses = [
            fitness_function(route, connections, distances_CS, penalties, ev_capacity, distances_between_points) for
            route in population]
        print("Generation number: ", generation)

        # Update best route and fitness
        for i, (route, fitness) in enumerate(zip(population, fitnesses)):
            if fitness > best_fitness and check_validity(route, connections, distances_CS, ev_capacity,
                                                         distances_between_points):
                best_route = copy.deepcopy(route)
                best_fitness = fitness

        print("Best route + best fitness:", best_route, best_fitness)

        # Select parents, perform crossover and mutation
        selected_parents = tournament_selection(population, fitnesses, tournament_size=3)
        print("Selected parents:", selected_parents)
        new_population = []
        i = 1
        while len(new_population) < population_size - 2:
            parent1 = selected_parents[i % len(selected_parents)]
            parent2 = selected_parents[(i + 1) % len(selected_parents)]
            i += 2
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate, points_with_ids))
            new_population.append(mutate(child2, mutation_rate, points_with_ids))

        print("Best route + best fitness after copying:", best_route, best_fitness)
        new_population.append(copy.deepcopy(best_route))  # Append a copy of the best route to the new population
        print("Best route added to new population:", best_route)
        population = new_population

        # Evaluate fitness of the new population
        fitnesses = [
            fitness_function(route, connections, distances_CS, penalties, ev_capacity, distances_between_points) for
            route in population]
        for i, (route, fitness) in enumerate(zip(population, fitnesses)):
            validity = check_validity(route, connections, distances_CS, ev_capacity, distances_between_points)
            print(f"Chromosome {i}: {route} with fitness {fitness} and validity {validity}")

    return best_route
