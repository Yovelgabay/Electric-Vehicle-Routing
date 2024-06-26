import random
import numpy as np
from scipy.spatial.distance import cdist


def assign_route_points_to_centroids(centroids, route):
    """
    Determine the closest centroid to each point on the route.

    Parameters:
    centroids (np.ndarray): Array of shape (k, 2) where k is the number of centroids and each centroid is represented as (x, y).
    route (np.ndarray): Array of shape (n, 2) where n is the number of points in the route and each point is represented as (x, y).

    Returns:
    list: A list of the index of the closest centroid for each point in the route.
    """
    closest_centroids = []

    for point in route:
        distances = np.linalg.norm(centroids - point, axis=1)  # Calculate Euclidean distance to each centroid
        closest_centroid_idx = np.argmin(distances)  # Get the index of the closest centroid
        closest_centroids.append(closest_centroid_idx)

    return closest_centroids


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


