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


def generate_route_with_checkpoints(num_points, scale, turn_amplitude, seed, checkpoints):
    """
    Generates a route with specified parameters and includes checkpoints.

    Parameters:
    - num_points (int): Number of points to generate along the route.
    - scale (float): Scaling factor for the route dimensions.
    - turn_amplitude (float): Amplitude of the zigzag turns.
    - seed (int): Random seed for reproducibility.
    - checkpoints (list): List of checkpoints that the route must pass through.

    Returns:
    - route (np.ndarray): Array of (x, y) coordinates defining the route.
    """
    np.random.seed(seed)
    random.seed(seed)
    start_point = np.array([0, 0])
    end_point = np.array([scale, scale])

    # Combine start point, checkpoints, and end point
    all_points = [start_point] + checkpoints + [end_point]

    waypoints = []
    for i in range(len(all_points) - 1):
        segment_start = all_points[i]
        segment_end = all_points[i + 1]
        segment_waypoints = np.linspace(segment_start, segment_end, num=int(num_points / (len(all_points) - 1)),
                                        endpoint=False)[1:]

        for j in range(len(segment_waypoints)):
            angle = np.pi / 2
            direction = random.choice([-1, 1])
            dx = turn_amplitude * direction * np.cos(angle)
            dy = turn_amplitude * direction * np.sin(angle)
            segment_waypoints[j] += np.array([dx, dy])

        waypoints.extend(segment_waypoints)

    waypoints.append(end_point)
    route = np.vstack([start_point, waypoints])

    return route


def calculate_route_points_distances(route):
    """
    Calculates distances between consecutive points on the route.

    Parameters:
    - route (np.ndarray): Array of (x, y) coordinates defining the route.

    Returns:
    - route_points_distances (np.ndarray): Distances between consecutive points on the route.
    """
    route_points_distances = np.linalg.norm(np.diff(route, axis=0), axis=1)
    return route_points_distances


def generate_random_charging_stations_and_queueing_time(seed, num_points, scale, route):
    """
    Generates random charging_stations and queueing_time.

    Parameters:
    - seed (int): Random seed for reproducibility.
    - num_points (int): Number of charging_stations to generate.
    - scale (float): Scaling factor for the point coordinates.
    - route (np.ndarray): Array of (x, y) coordinates defining the route.

    Returns:
    - points_with_ids (list): List of tuples containing point ID, coordinates, and closest route index.
    - queueing_time (np.ndarray): Array of queueing_time associated with each point.
    """
    np.random.seed(seed)
    charging_stations = np.random.rand(num_points, 2) * scale
    np.random.seed(seed + 1)
    queueing_time = np.random.uniform(1, 20, num_points)
    intersections = get_intersection_points(route, charging_stations)
    station_ids = sort_stations_by_route(intersections)
    points_with_ids = [(i, charging_stations[i], intersections[i][2]) for i in station_ids]
    return points_with_ids, queueing_time


def closest_point(route, charging_station):
    """
    Finds the closest route point to a given charging station.

    Parameters:
    - route (np.ndarray): Array of (x, y) coordinates defining the route.
    - point (np.ndarray): Coordinates of the point to find the closest route point for.

    Returns:
    - int: Index of the closest point on the route.
    """
    distances = cdist([charging_station], route, 'euclidean')
    return np.argmin(distances)


def get_intersection_points(route, charging_stations):
    """
    Determines the closest point on the route for each charging station.

    Parameters:
    - route (np.ndarray): Array of (x, y) coordinates defining the route.
    - charging_stations (np.ndarray): Array of (x, y) coordinates of the charging_stations.

    Returns:
    - intersections (list): List of tuples containing charging station index,
     coordinates, and closest route point index.
    """
    intersections = []
    for idx, charging_station in enumerate(charging_stations):
        closest_idx = closest_point(route, charging_station)
        intersections.append((idx, charging_station, closest_idx))
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


