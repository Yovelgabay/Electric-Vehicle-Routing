import numpy as np
import random
from scipy.spatial.distance import cdist


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return np.hypot(px - x1, py - y1)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    closest_x, closest_y = x1 + t * dx, y1 + t * dy
    return np.hypot(px - closest_x, py - closest_y)


def calculate_distances(points_with_ids, route):
    distances = np.zeros((len(points_with_ids), len(route) - 1))
    for i, (_, (px, py)) in enumerate(points_with_ids):  # Unpack the point
        for j in range(len(route) - 1):
            x1, y1 = route[j]
            x2, y2 = route[j + 1]
            distances[i, j] = point_to_segment_distance(px, py, x1, y1, x2, y2)
    min_distances = distances.min(axis=1)
    return min_distances


def fitness_function(route, connections, distances, penalties):
    if len(route) != len(set(route)):
        return 0.0000000001  # Return a very small value if there are duplicate stops
    # Initialize a new list to hold the results
    newList = []
    # Iterate over each connection tuple in the connections list
    for point in route:
        newList.append(connections[point][1])
    if newList != sorted(newList):
        return 0.0000000001
    total_distance = sum(distances[route[i]] for i in range(len(route)))
    # total_penalty = sum(penalties[route[i]] for i in range(len(route)))
    print(f"Total distance = {total_distance}")
    return 1 / total_distance if total_distance > 0 else 0.000000000001


def selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    return population[np.random.choice(len(population), p=probabilities)]


def crossover(parent1, parent2):
    size = min(len(parent1), len(parent2))
    if size < 2:
        return parent1, parent2
    cx_point = random.randint(1, size - 1)
    child1 = parent1[:cx_point] + parent2[cx_point:]
    child2 = parent2[:cx_point] + parent1[cx_point:]
    return child1, child2


def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route


def initialize_population(points_with_ids, population_size):
    population = []
    num_points = len(points_with_ids)
    for i in range(population_size):
        num_stops = 5
        stops = sorted(random.sample(range(num_points), num_stops))
        population.append(stops)
        print(f"chromosome {i}", stops)
    return population


def genetic_algorithm(points_with_ids, route, connections, population_size, generations, mutation_rate, penalties):
    print("initial population")
    population = initialize_population(points_with_ids, population_size)
    distances = calculate_distances(points_with_ids, route)
    best_route = None
    best_fitness = float('-inf')

    for generation in range(generations):
        fitnesses = [fitness_function(route, connections, distances, penalties) for route in population]
        new_population = []
        print("Generation number: ", generation)

        # Add the best route from the previous generation to the new population
        if best_route is not None:
            new_population.append(best_route)

        # Select parents, perform crossover and mutation
        for _ in range(population_size // 2 - 1):  # Minus 1 for the best route from the previous generation
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            print("chromosome ", child1)
            print("chromosome ", child2)
            new_population.extend([child1, child2])

        population = new_population

        # Evaluate fitness of the new population
        fitnesses = [fitness_function(route, connections, distances, penalties) for route in population]
        best_route_index = np.argmax(fitnesses)
        if fitnesses[best_route_index] > best_fitness:
            best_fitness = fitnesses[best_route_index]
            best_route = population[best_route_index]

    return best_route
