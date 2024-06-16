import numpy as np
import random
import copy

from scipy.spatial.distance import cdist


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return np.hypot(px - x1, py - y1)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    closest_x, closest_y = x1 + t * dx, y1 + t * dy
    return np.hypot(px - closest_x, py - closest_y)


def calculate_distances_of_CS(points_with_ids, route):
    distances = np.zeros((len(points_with_ids), len(route) - 1))
    for i, (_, (px, py), _) in enumerate(points_with_ids):  # Unpack the point
        for j in range(len(route) - 1):
            x1, y1 = route[j]
            x2, y2 = route[j + 1]
            distances[i, j] = point_to_segment_distance(px, py, x1, y1, x2, y2)
    min_distances = distances.min(axis=1)
    return min_distances


def check_validity(chromosome, connections, distances, ev_capacity, route_distances):
    total_distance = 0
    #print(f"Checking chromosome: {chromosome}")

    # Calculate the distance from the start point to the first charging station
    start_to_first = sum(route_distances[:connections[chromosome[0]][1]])  # Sum of route distances to the first charging station
    start_to_first += distances[chromosome[0]]  # Distance from the route to the first charging station
    if start_to_first > ev_capacity:
        #print(f"Invalid chromosome: start to first charging station distance {start_to_first} exceeds EV capacity")
        return False
    #print(f"start_to_first: {start_to_first}")
    total_distance += start_to_first

    for i in range(len(chromosome) - 1):
        start_station = chromosome[i]
        end_station = chromosome[i + 1]

        # Distance to drive from start_station to exit point on the route
        dist_to_route = distances[start_station]

        # Distance to drive on the route from exit point of start_station to exit point of end_station
        route_distance = sum(route_distances[connections[start_station][1]:connections[end_station][1]])

        # Distance to drive from exit point on the route to end_station
        dist_from_route = distances[end_station]

        segment_distance = dist_to_route + route_distance + dist_from_route

        #print(f"Segment {i}: start={start_station}, end={end_station}, dist_to_route={dist_to_route}, route_distance={route_distance}, dist_from_route={dist_from_route}, segment_distance={segment_distance}")

        if segment_distance > ev_capacity:
            #print("Invalid chromosome: exceeds EV capacity")
            return False

        total_distance += segment_distance

    # Calculate the distance from the last charging station to the destination
    last_station_idx = chromosome[-1]
    route_to_end = sum(route_distances[connections[last_station_idx][1]:])
    last_to_end = route_to_end + distances[last_station_idx]  # Distance from the last charging station to the end of the route
    if last_to_end > ev_capacity:
        #print(f"Invalid chromosome: last charging station to destination distance {last_to_end} exceeds EV capacity")
        return False
    #print(f"last_to_end: {last_to_end}")

    total_distance += last_to_end
    #print("Valid chromosome")
    return True



def fitness_function(chromosome, connections, distances, penalties, ev_capacity, route_distances):
    if not check_validity(chromosome, connections, distances, ev_capacity, route_distances):
        return 0.0000000001  # Return a fitness of 0 if the chromosome is not valid

    total_distance = sum(distances[stop] for stop in chromosome)
    total_penalty = sum(penalties[stop] for stop in chromosome)
    stop_penalty = len(chromosome)
    return 1 / (total_distance + total_penalty + stop_penalty) if total_distance + total_penalty + stop_penalty > 0 else 0.000000000001


def tournament_selection(population, fitnesses, tournament_size=2):
    selected_parents = []
    indices = list(range(len(population)))

    while indices:
        # Randomly select a tournament group
        tournament_indices = random.sample(indices, min(tournament_size, len(indices)))
        tournament_fitnesses = [fitnesses[idx] for idx in tournament_indices]

        # Select the best individual from the tournament
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        selected_parents.append(population[winner_idx])

        # Remove the selected individual from the pool to avoid duplication
        indices.remove(winner_idx)

    return selected_parents


def crossover(parent1, parent2):
    size1, size2 = len(parent1), len(parent2)
    # Create sets to find common nodes
    set1 = set(parent1)
    set2 = set(parent2)
    common_nodes = list(
        set1 & set2 - set([parent1[0], parent1[-1], parent2[0], parent2[-1]]))  # Exclude source and destination

    if not common_nodes:
        return parent1, parent2  # No common genes, return the parents unchanged

    # Randomly choose one of the common nodes as the crossing site
    crossing_node = random.choice(common_nodes)

    # Find indices of the crossing node in both parents
    idx1 = parent1.index(crossing_node)
    idx2 = parent2.index(crossing_node)

    # Create offspring by slicing at the crossing node and swapping segments
    child1_part1 = parent1[:idx1]
    child1_part2 = parent2[idx2:]
    child1 = child1_part1 + child1_part2

    child2_part1 = parent2[:idx2]
    child2_part2 = parent1[idx1:]
    child2 = child2_part1 + child2_part2

    return child1, child2


def mutate(route, mutation_rate, points_with_ids):
    if random.random() < mutation_rate:
        # Select a random locus to mutate
        idx = random.randint(0, len(route) - 1)

        # Determine the valid range for the new station at this locus
        if idx == 0:
            min_val = 0  # No restriction from the left
        else:
            min_val = route[idx - 1]

        if idx == len(route) - 1:
            max_val = len(points_with_ids) - 1  # No restriction from the right
        else:
            max_val = route[idx + 1]

        # Choose a new station within the valid range that is different from the current station
        possible_stations = [i for i in range(min_val + 1, max_val) if i != route[idx]]

        if possible_stations:
            new_station = random.choice(possible_stations)
            route[idx] = new_station

    return route


def initialize_population(points_with_ids, population_size):
    population = []
    num_points = len(points_with_ids)
    for i in range(population_size):
        num_stops = random.randint(1, num_points)  # Vary the number of stops
        stops = sorted(random.sample(range(num_points), num_stops))
        population.append(stops)
        print(f"chromosome {i}", stops)
    return population


import copy


def genetic_algorithm(points_with_ids, route, connections, population_size, generations, mutation_rate, penalties,
                      ev_capacity, distances_between_points):
    print("initial population")
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

        print("best_route + best_fitness 1", best_route, best_fitness)

        # Select parents, perform crossover and mutation
        selected_parents = tournament_selection(population, fitnesses, tournament_size=3)
        print("selected_parents", selected_parents)
        new_population = []
        i = 1
        while len(new_population) < population_size - 2:
            parent1 = selected_parents[i % len(selected_parents)]
            parent2 = selected_parents[(i + 1) % len(selected_parents)]
            i += 2
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate, points_with_ids))
            new_population.append(mutate(child2, mutation_rate, points_with_ids))

        print("best_route + best_fitness copy", best_route, best_fitness)
        new_population.append(copy.deepcopy(best_route))  # Append a copy of the best route to the new population
        print(best_route, "added to new_population")
        population = new_population

        # Evaluate fitness of the new population
        fitnesses = [
            fitness_function(route, connections, distances_CS, penalties, ev_capacity, distances_between_points) for
            route in population]
        for i, (route, fitness) in enumerate(zip(population, fitnesses)):
            validity = check_validity(route, connections, distances_CS, ev_capacity, distances_between_points)
            print(f"chromosome {i}: {route} with fitness {fitness} and validity {validity}")

    return best_route

