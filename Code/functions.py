import random
import numpy as np
from scipy.spatial.distance import cdist

from Code.parameters import MIN_QUEUEING_TIME, MAX_QUEUEING_TIME


def assign_route_points_to_centroids(centroids, route):
    """
    Determine the closest centroid to each point on the route.
    """

    closest_centroids = []
    # Iterate over each point in the route
    for point in route:
        # Calculate Euclidean distance from the point to each centroid
        distances = np.linalg.norm(centroids - point, axis=1)
        # Find the index of the closest centroid
        closest_centroid_idx = np.argmin(distances)
        # Append the index of the closest centroid to the result list
        closest_centroids.append(closest_centroid_idx)

    return closest_centroids


def generate_route_with_checkpoints(num_points, scale, turn_amplitude, seed, checkpoints):
    """
    Generates a route with specified parameters and includes checkpoints.
    """

    np.random.seed(seed)
    random.seed(seed)

    # Define start and end points of the route
    start_point = np.array([0, 0])
    end_point = np.array([scale, scale])

    # Combine start point, checkpoints, and end point
    all_points = [start_point] + checkpoints + [end_point]

    waypoints = []
    # Generate waypoints between each pair of points
    for i in range(len(all_points) - 1):
        segment_start = all_points[i]
        segment_end = all_points[i + 1]
        segment_waypoints = np.linspace(segment_start, segment_end, num=int(num_points / (len(all_points) - 1)),
                                        endpoint=False)[1:]

        # Add zigzag turns to the waypoints
        for j in range(len(segment_waypoints)):
            angle = np.pi / 2
            direction = random.choice([-1, 1])
            dx = turn_amplitude * direction * np.cos(angle)
            dy = turn_amplitude * direction * np.sin(angle)
            segment_waypoints[j] += np.array([dx, dy])

        waypoints.extend(segment_waypoints)

    # Append the end point to the route
    waypoints.append(end_point)
    route = np.vstack([start_point, waypoints])

    return route


def calculate_route_points_distances(route):
    """
    Calculates distances between consecutive points on the route.
    """

    # Calculate Euclidean distances between consecutive route points
    route_points_distances = np.linalg.norm(np.diff(route, axis=0), axis=1)
    return route_points_distances


def generate_random_charging_stations_and_queueing_time(seed, num_points, scale, route):
    """
    Generates random charging stations and queueing times.
    """

    np.random.seed(seed)
    # Generate random coordinates for the charging stations
    charging_stations = np.random.rand(num_points, 2) * scale

    # Generate random queueing times for each charging station
    np.random.seed(seed + 1)
    queueing_time = np.random.uniform(MIN_QUEUEING_TIME, MAX_QUEUEING_TIME, num_points)

    # Determine intersection points between route and charging stations
    intersections = get_intersection_points(route, charging_stations)

    # Sort stations based on their proximity to the route
    station_ids = sort_stations_by_route(intersections)

    # Create list of tuples with station ID, coordinates, and closest route index
    points_with_ids = [(i, charging_stations[i], intersections[i][2]) for i in station_ids]
    return points_with_ids, queueing_time


def closest_point(route, charging_station):
    """
    Finds the closest route point to a given charging station.
    """

    # Calculate Euclidean distance between the charging station and each point on the route
    distances = cdist([charging_station], route, 'euclidean')
    # Find the index of the closest point on the route
    return np.argmin(distances)


def get_intersection_points(route, charging_stations):
    """
    Determines the closest point on the route for each charging station.
    """

    intersections = []
    # Iterate over each charging station
    for idx, charging_station in enumerate(charging_stations):
        # Find the closest route point for the charging station
        closest_idx = closest_point(route, charging_station)
        # Append the intersection details to the list
        intersections.append((idx, charging_station, closest_idx))
    return intersections


def sort_stations_by_route(intersections):
    """
    Sorts stations based on their proximity to the route.
    """

    # Sort the intersections by the index of the closest route point
    intersections.sort(key=lambda x: x[2])
    # Return the sorted list of original indices
    return [x[0] for x in intersections]
