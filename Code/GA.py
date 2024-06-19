import numpy as np
import random
import copy

from scipy.spatial.distance import cdist


def point_to_point_distance(px, py, qx, qy):
    """
    Calculates the distance between two points (px, py) and (qx, qy).

    Parameters:
    - px (float): X-coordinate of the first point.
    - py (float): Y-coordinate of the first point.
    - qx (float): X-coordinate of the second point.
    - qy (float): Y-coordinate of the second point.

    Returns:
    - distance (float): The Euclidean distance between the two points.
    """
    return np.hypot(px - qx, py - qy)


def point_to_closest_point_distance(px, py, route):
    """
    Calculates the shortest distance from a point (px, py) to the closest point on the route.

    Parameters:
    - px (float): X-coordinate of the point.
    - py (float): Y-coordinate of the point.
    - route (np.ndarray): Array of route waypoints with shape (n, 2), where each row is [x, y].

    Returns:
    - min_distance (float): The shortest distance from the point to the closest point on the route.
    """
    min_distance = float('inf')
    for (rx, ry) in route:
        distance = np.hypot(px - rx, py - ry)
        if distance < min_distance:
            min_distance = distance
    return min_distance


def calculate_distances_of_CS(points_with_ids, route):
    """
    Calculates the minimum distance from each charging station to the nearest point on the route.

    Parameters:
    - points_with_ids (list): List of tuples (index, coordinates, closest route segment index) for each charging station.
    - route (np.ndarray): Array of route waypoints.

    Returns:
    - min_distances (np.ndarray): Array of the shortest distances from each charging station to the closest point on the route.
    """
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


def calculate_exceeded_kilometers(route, connections, distances, ev_capacity, route_distances):
    """
    Calculates how many kilometers the given route exceeded the EV capacity in total.

    Parameters:
    - route (list): List of indices representing the route of the EV through charging stations.
    - connections (list): List of tuples defining connections between points and route segments.
    - distances (np.ndarray): Array of distances from charging stations to the route.
    - ev_capacity (float): Maximum distance the EV can travel on a single charge.
    - route_distances (list): List of distances between consecutive waypoints on the route.

    Returns:
    - total_excess_km (float): The total kilometers by which the route exceeded the EV capacity.
    """
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
    exceeded_km = 0
    if not check_validity(chromosome, connections, distances, ev_capacity, route_distances):
        exceeded_km = 50 * calculate_exceeded_kilometers(chromosome, connections, distances, ev_capacity,
                                                        route_distances)
        print(f"Exceeded chromosome: {chromosome}, {exceeded_km}", )

    total_distance = sum(distances[stop] for stop in chromosome)
    # total_penalty = sum(penalties[stop] for stop in chromosome)
    total_penalty = 0
    stop_penalty = len(chromosome) * 50  # each stop added X km to the total_distance
    return 1 / (total_distance + total_penalty + stop_penalty + exceeded_km) \
        if total_distance + total_penalty + stop_penalty + exceeded_km > 0 else 0.000000000001


def tournament_selection(population, fitnesses, tournament_size):
    """
    Selects one parent for crossover based on a tournament selection method.

    Parameters:
    - population (list): List of routes (chromosomes) in the current generation.
    - fitnesses (list): List of fitness values corresponding to the routes in the population.
    - tournament_size (int): Size of the tournament group for selection.

    Returns:
    - selected_parent (list): The selected parent route for crossover.
    """
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
            if fitness > best_fitness:
                best_route = copy.deepcopy(route)
                best_fitness = fitness

        print("Best route + best fitness:", best_route, best_fitness)

        # Select parents, perform crossover and mutation
        new_population = []
        while len(new_population) < population_size - 2:
            parent1 = tournament_selection(population, fitnesses, tournament_size=4)
            parent2 = tournament_selection(population, fitnesses, tournament_size=4)

            print("Selected parents:", parent1, parent2)

            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate, points_with_ids))
            new_population.append(mutate(child2, mutation_rate, points_with_ids))

        '''
        print("Best route + best fitness after copying:", best_route, best_fitness)
        new_population.append(copy.deepcopy(best_route))  # Append a copy of the best route to the new population
        print("Best route added to new population:", best_route)
        '''
        population = new_population

        # Evaluate fitness of the new population
        fitnesses = [
            fitness_function(route, connections, distances_CS, penalties, ev_capacity, distances_between_points) for
            route in population]
        for i, (route, fitness) in enumerate(zip(population, fitnesses)):
            validity = check_validity(route, connections, distances_CS, ev_capacity, distances_between_points)
            print(f"Chromosome {i}: {route} with fitness {fitness} and validity {validity}")

    return best_route
