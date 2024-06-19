import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from GA import genetic_algorithm
from kmeans import kmeans_clustering
from parameters import *


def generate_zigzag_route(num_points, scale, turn_amplitude, seed):
    """
    Generates a zigzag route with specified parameters.

    Parameters:
    - num_points (int): Number of points to generate along the route.
    - scale (float): Scaling factor for the route dimensions.
    - turn_amplitude (float): Amplitude of the zigzag turns.
    - seed (int): Random seed for reproducibility.

    Returns:
    - route (np.ndarray): Array of (x, y) coordinates defining the route.
    """
    np.random.seed(seed)
    random.seed(seed)
    start_point = np.array([0, 0])
    end_point = np.array([scale, scale])
    waypoints = np.linspace(start_point, end_point, num=num_points - 1)[1:-1]
    for i in range(len(waypoints)):
        angle = np.pi / 2
        direction = random.choice([-1, 1])
        dx = turn_amplitude * direction * np.cos(angle)
        dy = turn_amplitude * direction * np.sin(angle)
        waypoints[i] += np.array([dx, dy])
    route = np.vstack([start_point, waypoints, end_point])

    return route


def calculate_distances_between_points(route):
    """
    Calculates distances between consecutive points on the route.

    Parameters:
    - route (np.ndarray): Array of (x, y) coordinates defining the route.

    Returns:
    - distances_between_points (np.ndarray): Distances between consecutive points on the route.
    """
    distances_between_points = np.linalg.norm(np.diff(route, axis=0), axis=1)
    return distances_between_points


def generate_random_points_and_penalties(seed, num_points, scale, route):
    """
    Generates random points and penalties.

    Parameters:
    - seed (int): Random seed for reproducibility.
    - num_points (int): Number of points to generate.
    - scale (float): Scaling factor for the point coordinates.
    - route (np.ndarray): Array of (x, y) coordinates defining the route.

    Returns:
    - points_with_ids (list): List of tuples containing point ID, coordinates, and closest route index.
    - penalties (np.ndarray): Array of penalties associated with each point.
    """
    np.random.seed(seed)
    points = np.random.rand(num_points, 2) * scale
    np.random.seed(seed + 1)
    penalties = np.random.uniform(1, 20, num_points)
    intersections = get_intersection_points(route, points)
    station_ids = sort_stations_by_route(intersections)
    points_with_ids = [(i, points[i], intersections[i][2]) for i in station_ids]
    return points_with_ids, penalties


def visualize_route(points, route, title, penalties, connections=[], distances=[]):
    """
    Visualizes the route and charging stations.

    Parameters:
    - points (np.ndarray): Array of (x, y) coordinates of the points.
    - route (np.ndarray): Array of (x, y) coordinates defining the route.
    - title (str): Title of the plot.
    - penalties (np.ndarray): Array of penalties associated with each point.
    - connections (list): List of tuples defining connections between points and route segments.
    - distances (list): List of distances between consecutive route points.

    Displays:
    - A plot showing the route, points, penalties, and connections.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(route[:, 0], route[:, 1], 'r-o', label='Route')
    plt.scatter(points[:, 0], points[:, 1], color='blue', label='Charging Stations')
    for i, (x, y) in enumerate(points):
        plt.text(x + 0.5, y + 0.5, f'{i}', fontsize=12, color='green')

    for i in range(len(route) - 1):
        mid_x = (route[i, 0] + route[i + 1, 0]) / 2
        mid_y = (route[i, 1] + route[i + 1, 1]) / 2
        plt.text(mid_x, mid_y, f'{distances[i]:.2f}', fontsize=10, color='black')

    for start, end in connections:
        plt.plot([points[start][0], route[end][0]], [points[start][1], route[end][1]], 'k--')
        plt.text((points[start][0] + route[end][0]) / 2, (points[start][1] + route[end][1]) / 2, f'{start},{end}',
                 fontsize=8, color='purple')
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()
    plt.show()


def closest_point(route, point):
    """
    Finds the closest point on the route to the given point.

    Parameters:
    - route (np.ndarray): Array of (x, y) coordinates defining the route.
    - point (np.ndarray): Coordinates of the point to find the closest route point for.

    Returns:
    - int: Index of the closest point on the route.
    """
    distances = cdist([point], route, 'euclidean')
    return np.argmin(distances)


def get_intersection_points(route, points):
    """
    Determines the closest points on the route for each point.

    Parameters:
    - route (np.ndarray): Array of (x, y) coordinates defining the route.
    - points (np.ndarray): Array of (x, y) coordinates of the points.

    Returns:
    - intersections (list): List of tuples containing point index, coordinates, and closest route point index.
    """
    intersections = []
    for idx, point in enumerate(points):
        closest_idx = closest_point(route, point)
        intersections.append((idx, point, closest_idx))
    return intersections


def sort_stations_by_route(intersections):
    """
    Sorts stations based on their proximity to the route.

    Parameters:
    - intersections (list): List of tuples containing point index, coordinates, and closest route point index.

    Returns:
    - list: List of original indices of points sorted by their closest route point index.
    """
    intersections.sort(key=lambda x: x[2])
    return [x[0] for x in intersections]


ev_capacity = EV_CAPACITY
# Generating data
num_route_points = NUM_ROUTE_POINTS
route = generate_zigzag_route(num_route_points, 100, 10, seed=53)
distances_between_points = calculate_distances_between_points(route)
points_matrix, penalties = generate_random_points_and_penalties(13, 100, 100, route)

# Extract points from points_matrix
points = np.array([point[1] for point in points_matrix])

# Visualize initial setup
intersections = get_intersection_points(route, points)
connections = [(idx, closest_point(route, pt)) for idx, pt, _ in intersections]
visualize_route(points, route, 'Zigzag Route Visualization', penalties, connections, distances_between_points)

# Apply clustering and genetic algorithm
kmeans_clustering(points)
best_route_indices = genetic_algorithm(points_matrix, route, connections, population_size=POPULATION_SIZE,
                                       generations=GENERATIONS, mutation_rate=MUTATION_RATE,
                                       penalties=penalties, ev_capacity=ev_capacity,
                                       distances_between_points=distances_between_points)
print("Best Route:", best_route_indices)


def demonstrate_chosen_route(route, points, best_route_indices, connections, title, distances):
    """
    Visualizes the chosen route and charging stations.

    Parameters:
    - route (np.ndarray): Array of (x, y) coordinates defining the route.
    - points (np.ndarray): Array of (x, y) coordinates of the points.
    - best_route_indices (list): List of indices representing the best route.
    - connections (list): List of tuples defining connections between points and route segments.
    - title (str): Title of the plot.
    - distances (list): List of distances between consecutive route points.

    Displays:
    - A plot showing the route, chosen charging stations, and connections.
    """
    plt.figure(figsize=(10, 8))

    # Highlight the chosen charging stations.
    chosen_points = points[best_route_indices]
    plt.scatter(chosen_points[:, 0], chosen_points[:, 1], color='blue', s=100, zorder=5,
                label='Chosen Charging Stations')

    for i, (x, y) in enumerate(chosen_points):
        plt.text(x + 0.5, y + 0.5, f'{best_route_indices[i]}', fontsize=12, color='gold')

    # Plot connections only to the chosen stations
    chosen_connections = [(idx, closest_point(route, points[idx])) for idx in best_route_indices]
    for start, end in chosen_connections:
        plt.plot([route[end][0], points[start][0]], [route[end][1], points[start][1]], 'r-', linewidth=2)
        mid_x = (route[end][0] + points[start][0]) / 2
        mid_y = (route[end][1] + points[start][1]) / 2
        segment_length = np.linalg.norm(route[end] - points[start])
        plt.text(mid_x, mid_y, f'{segment_length:.2f}', fontsize=10, color='pink')

    # Plot the entire route and annotate all segment lengths
    plt.scatter(route[:, 0], route[:, 1], color='black', alpha=0.5, label='Route Waypoints')
    plt.plot(route[:, 0], route[:, 1], 'green', linestyle='dashed', alpha=0.5)

    for i in range(len(route) - 1):
        mid_x = (route[i, 0] + route[i + 1, 0]) / 2
        mid_y = (route[i, 1] + route[i + 1, 1]) / 2
        plt.text(mid_x, mid_y, f'{distances[i]:.2f}', fontsize=10, color='black')

    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()
    plt.show()


demonstrate_chosen_route(route, points, best_route_indices, connections, 'Chosen Route Visualization',
                         distances_between_points)
