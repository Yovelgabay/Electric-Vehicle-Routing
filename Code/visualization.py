import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def plot_centroids_and_route(centroids, route, closest_centroids):
    """
    Visualize centroids with the corresponding route.

    Parameters:
    centroids (np.ndarray): Array of shape (k, 2) where k is the number of centroids and each centroid is represented as (x, y).
    route (np.ndarray): Array of shape (n, 2) where n is the number of points in the route and each point is represented as (x, y).
    closest_centroids (list): List of indices of the closest centroids for each point in the route.
    """
    # Define colors for centroids and corresponding route points
    colors = plt.cm.get_cmap('tab10', len(centroids))

    plt.figure(figsize=(10, 6))

    # Plot centroids
    for idx, centroid in enumerate(centroids):
        plt.scatter(*centroid, color=colors(idx), label=f'Centroid {idx}', edgecolor='black', s=200, marker='X')

    # Plot route points with colors corresponding to closest centroids
    for idx, (point, centroid_idx) in enumerate(zip(route, closest_centroids)):
        plt.scatter(*point, color=colors(centroid_idx), edgecolor='black')
        plt.plot([point[0], centroids[centroid_idx][0]], [point[1], centroids[centroid_idx][1]],
                 color=colors(centroid_idx), linestyle='dotted')

    # Plot route lines connecting the points
    plt.plot(route[:, 0], route[:, 1], color='grey', linestyle='-', linewidth=1, label='Route')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Centroids and Route Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_route(points, route, title, penalties, connections=None, distances=None):
    """
    Visualizes the route and charging stations with penalties color mapping.

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
    if distances is None:
        distances = []
    if connections is None:
        connections = []

    plt.figure(figsize=(10, 8))

    # Plot the route
    plt.plot(route[:, 0], route[:, 1], 'r-o', label='Route')

    # Normalize penalties for color mapping
    norm = Normalize(penalties.min(), penalties.max())
    cmap = cm.get_cmap('viridis')
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Scatter points with color mapped based on penalties
    sc = plt.scatter(points[:, 0], points[:, 1], c=penalties, cmap=cmap, norm=norm, label='Charging Stations',
                     edgecolor='black')

    # Add colorbar to indicate penalty scale
    cbar = plt.colorbar(sm)
    cbar.set_label('Penalties')

    # Annotate each point
    for i, (x, y) in enumerate(points):
        plt.text(x + 0.5, y + 0.5, f'{i}', fontsize=12, color='black')

    # Annotate distances between route points if distances are provided
    if len(distances) > 0:
        for i in range(len(route) - 1):
            mid_x = (route[i, 0] + route[i + 1, 0]) / 2
            mid_y = (route[i, 1] + route[i + 1, 1]) / 2
            plt.text(mid_x, mid_y, f'{distances[i]:.2f}', fontsize=10, color='black')

    # Plot connections
    for start, end in connections:
        plt.plot([points[start][0], route[end][0]], [points[start][1], route[end][1]], 'k--')

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
