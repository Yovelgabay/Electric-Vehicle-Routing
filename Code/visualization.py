import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
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

    # Plot route points with colors corresponding to the closest centroids
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


def visualize_route(charging_stations, route, title, queueing_time, connections=None, distances=None,
                    points_diff=0, route_diff=0):
    """
    Visualizes the route and charging stations with queueing_time color mapping.

    Parameters:
    - charging_stations (np.ndarray): Array of (x, y) coordinates of the charging_stations.
    - route (np.ndarray): Array of (x, y) coordinates defining the route.
    - title (str): Title of the plot.
    - queueing_time (np.ndarray): Array of queueing_time associated with each point.
    - connections (list): List of tuples defining connections between charging_stations and route segments.
    - distances (list): List of distances between consecutive route points.

    Displays:
    - A plot showing the route, charging_stations, queueing_time, and connections.
    """
    if distances is None:
        distances = []
    if connections is None:
        connections = []

    plt.figure(figsize=(10, 8))

    # Plot the route
    plt.plot(route[:, 0], route[:, 1], 'r-o', label='Route')

    # Define custom colormap ranging from green to red
    cmap = plt.cm.get_cmap('RdYlGn_r')  # Reversed RdYlGn colormap

    # Normalize queueing_time for color mapping
    norm = Normalize(vmin=queueing_time.min(), vmax=queueing_time.max())

    # Scatter charging_stations with color mapped based on queueing_time
    sc = plt.scatter(charging_stations[:, 0], charging_stations[:, 1], c=queueing_time, cmap=cmap,
                     norm=norm, label='Charging Stations', edgecolor='black')

    # Add colorbar to indicate penalty scale
    cbar = plt.colorbar(sc)
    cbar.set_label('queueing_time')

    # Annotate each point
    for i, (x, y) in enumerate(charging_stations):
        plt.text(x + 0.5, y + 0.5, f'{i}', fontsize=12, color='black')

    # Annotate distances between route points if distances are provided
    if len(distances) > 0:
        for i in range(len(route) - 1):
            mid_x = (route[i, 0] + route[i + 1, 0]) / 2
            mid_y = (route[i, 1] + route[i + 1, 1]) / 2
            plt.text(mid_x, mid_y, f'{distances[i]:.2f}', fontsize=10, color='black')

    # Plot connections
    for start, end in connections:
        plt.plot([charging_stations[start - points_diff][0], route[end - route_diff][0]],
                 [charging_stations[start - points_diff][1], route[end - route_diff][1]], 'k--')

    # Set the limits of the plot
    plt.xlim(-2, 102)
    plt.ylim(-2, 102)

    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()
    plt.show()


def closest_point(route, charging_station):
    """
    Finds the closest point on the route to the given point.

    Parameters:
    - route (np.ndarray): Array of (x, y) coordinates defining the route.
    - point (np.ndarray): Coordinates of the point to find the closest route point for.

    Returns:
    - int: Index of the closest point on the route.
    """
    distances = cdist([charging_station], route, 'euclidean')
    return np.argmin(distances)


def demonstrate_chosen_route(route, charging_stations, best_route_indices, connections, title, distances):
    """
    Visualizes the chosen route and charging stations.

    Parameters:
    - route (np.ndarray): Array of (x, y) coordinates defining the route.
    - charging_stations (np.ndarray): Array of (x, y) coordinates of the charging_stations.
    - best_route_indices (list): List of indices representing the best route.
    - connections (list): List of tuples defining connections between charging_stations and route segments.
    - title (str): Title of the plot.
    - distances (list): List of distances between consecutive route points.

    Displays:
    - A plot showing the route, chosen charging stations, and connections.
    """
    plt.figure(figsize=(10, 8))

    # Highlight the chosen charging stations.
    chosen_charging_stations = charging_stations[best_route_indices]
    plt.scatter(chosen_charging_stations[:, 0], chosen_charging_stations[:, 1], color='blue', s=100, zorder=5,
                label='Chosen Charging Stations')

    for i, (x, y) in enumerate(chosen_charging_stations):
        plt.text(x + 0.5, y + 0.5, f'{best_route_indices[i]}', fontsize=12, color='gold')

    # Plot connections only to the chosen stations
    chosen_connections = [(idx, closest_point(route, charging_stations[idx])) for idx in best_route_indices]
    for start, end in chosen_connections:
        plt.plot([route[end][0], charging_stations[start][0]], [route[end][1], charging_stations[start][1]], 'r-', linewidth=2)
        mid_x = (route[end][0] + charging_stations[start][0]) / 2
        mid_y = (route[end][1] + charging_stations[start][1]) / 2
        segment_length = np.linalg.norm(route[end] - charging_stations[start])
        plt.text(mid_x, mid_y, f'{segment_length:.2f}', fontsize=10, color='pink')

    # Plot the entire route and annotate all segment lengths
    plt.scatter(route[:, 0], route[:, 1], color='black', alpha=0.5, label='Route Waypoints')
    plt.plot(route[:, 0], route[:, 1], 'green', linestyle='dashed', alpha=0.5)

    for i in range(len(route) - 1):
        mid_x = (route[i, 0] + route[i + 1, 0]) / 2
        mid_y = (route[i, 1] + route[i + 1, 1]) / 2
        plt.text(mid_x, mid_y, f'{distances[i]:.2f}', fontsize=10, color='black')

    plt.xlim(-2, 102)
    plt.ylim(-2, 102)

    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()
    plt.show()


# Function to print waiting time for each stop
def print_waiting_times(best_charging_stations, charging_stations_matrix, labels, starting_point_cluster,
                        average_waiting_time, queueing_time):
    for stop_index in best_charging_stations:
        stop_id, (x, y), _ = charging_stations_matrix[stop_index]
        cluster = labels[stop_index]
        if cluster == starting_point_cluster:
            waiting_time = queueing_time[stop_index]  # Use real waiting time
        else:
            waiting_time = average_waiting_time  # Use average waiting time
        print(f"CS {stop_index}: Waiting Time = {waiting_time:.2f}")


def visualize_clustering(num_clusters, charging_stations, labels, centroids):
    # Visualize the clustered charging_stations
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap('tab10')

    for i in range(num_clusters):
        cluster_charging_stations = charging_stations[labels == i]
        plt.scatter(cluster_charging_stations[:, 0], cluster_charging_stations[:, 1], color=cmap(i), marker='o', label=f'Cluster {i + 1}')
        plt.scatter(centroids[i, 0], centroids[i, 1], color='black', marker='x', s=100, linewidths=3)

    plt.title('K-Means Clustering of Random Charging Stations')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

    # # Print charging_stations and their assigned clusters
    # for i, (point, label) in enumerate(zip(charging_stations, labels)):
    #     print(f'Point {i}: ({point[0]:.2f}, {point[1]:.2f}) - Cluster {label + 1}')


def print_segment_lengths(route, connections, best_charging_stations, charging_stations):
    # If there are no charging stations selected, print total route distance and return
    if not best_charging_stations:
        total_distance = sum(np.linalg.norm(route[i + 1] - route[i]) for i in range(len(route) - 1))
        print(f"Total route distance: {total_distance:.2f} (No charging stations needed)")
        return
    # Calculate distances from charging stations to the closest route points
    distances = []
    for stop_index in best_charging_stations:
        _, (x, y), _ = charging_stations[stop_index]
        closest_route_index = closest_point(route, (x, y))
        min_distance = np.hypot(x - route[closest_route_index][0], y - route[closest_route_index][1])
        distances.append(min_distance)

    segments_lengths = []
    # Calculate length from start to first charging station
    start_to_first = sum(
        np.linalg.norm(route[i + 1] - route[i]) for i in range(connections[best_charging_stations[0]][1]))
    segments_lengths.append(start_to_first)

    # Calculate lengths between consecutive charging stations
    for i in range(len(best_charging_stations) - 1):
        current_charging_station = best_charging_stations[i]
        next_charging_station = best_charging_stations[i + 1]

        closest_route_point_current = connections[current_charging_station][1]
        closest_route_point_next = connections[next_charging_station][1]

        segment_length = sum(np.linalg.norm(route[j + 1] - route[j]) for j in
                             range(closest_route_point_current, closest_route_point_next))
        segments_lengths.append(segment_length)

    # Calculate length from last charging station to endpoint
    last_charging_station = best_charging_stations[-1]
    endpoint_index = len(route) - 1
    last_to_end = sum(
        np.linalg.norm(route[i + 1] - route[i]) for i in range(connections[last_charging_station][1], endpoint_index))
    segments_lengths.append(last_to_end)

    # Print segment lengths with distances to/from charging stations
    for i in range(len(segments_lengths)):
        if i == 0:
            segments_lengths[i] += distances[i]
            print(f"Travel distance from the starting point to CS "
                  f"{best_charging_stations[i]} is: {segments_lengths[i]:.2f}")
        elif i != (len(segments_lengths) - 1):
            segments_lengths[i] += distances[i] + distances[i - 1]
            print(f"Travel distance from CS {best_charging_stations[i - 1]} to CS "
                  f"{best_charging_stations[i]} is: {segments_lengths[i]:.2f}")
        else:
            segments_lengths[i] += distances[i - 1]
            print(f"Travel distance from CS "
                  f"{best_charging_stations[i - 1]} to the endpoint is: {segments_lengths[i]:.2f}")


def update_plot(ax, route, charging_stations, best_charging_stations, connections, route_points_distances, queueing_time, generation):
    """
    Update the plot with the best route and charging stations at each generation.

    Parameters:
    - ax: Matplotlib axis object.
    - route: Array of (x, y) coordinates defining the route.
    - charging_stations: Array of (x, y) coordinates of the charging_stations
    - best_charging_stations: List of indices representing the best charging stations.
    - connections: List of tuples defining connections between charging_stations and route segments.
    - route_points_distances: List of distances between consecutive route points.
    - queueing_time: Array of penalty values for each charging station.
    - generation: Current generation number for display.
    """
    ax.clear()
    ax.plot(route[:, 0], route[:, 1], 'r-o', label='Route')

    # Define custom colormap ranging from green to red
    cmap = plt.cm.get_cmap('RdYlGn_r')  # Reversed RdYlGn colormap

    # Normalize queueing_time for color mapping
    norm = Normalize(vmin=min(queueing_time), vmax=max(queueing_time))

    # Get chosen charging_stations and their queueing_time
    chosen_charging_stations = charging_stations[best_charging_stations]
    chosen_queueing_time = queueing_time[best_charging_stations]

    # Scatter chosen charging_stations with color based on queueing_time
    sc = ax.scatter(chosen_charging_stations[:, 0], chosen_charging_stations[:, 1], c=cmap(norm(chosen_queueing_time)), s=100, zorder=5,
                    label='Chosen Charging Stations')

    for i, (x, y) in enumerate(chosen_charging_stations):
        ax.text(x + 0.5, y + 0.5, f'{best_charging_stations[i]}', fontsize=12, color='gold')

    chosen_connections = [(idx, closest_point(route, charging_stations[idx])) for idx in best_charging_stations]
    for start, end in chosen_connections:
        ax.plot([route[end][0], charging_stations[start][0]], [route[end][1], charging_stations[start][1]], 'r-', linewidth=2)
        mid_x = (route[end][0] + charging_stations[start][0]) / 2
        mid_y = (route[end][1] + charging_stations[start][1]) / 2
        segment_length = np.linalg.norm(route[end] - charging_stations[start])
        ax.text(mid_x, mid_y, f'{segment_length:.2f}', fontsize=10, color='pink')

    ax.scatter(route[:, 0], route[:, 1], color='black', alpha=0.5, label='Route Waypoints')
    ax.plot(route[:, 0], route[:, 1], 'green', linestyle='dashed', alpha=0.5)

    for i in range(len(route) - 1):
        mid_x = (route[i, 0] + route[i + 1, 0]) / 2
        mid_y = (route[i, 1] + route[i + 1, 1]) / 2
        ax.text(mid_x, mid_y, f'{route_points_distances[i]:.2f}', fontsize=10, color='black')

    # Add generation number text at the top middle
    ax.text(0.5, 0.95, f'Generation: {generation}', transform=ax.transAxes, fontsize=14,
            horizontalalignment='center', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)

    ax.set_title("Chosen Route Visualization")
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True)
    ax.legend()

    # Add a color bar to indicate penalty values
    if not hasattr(ax, 'cbar') or ax.cbar is None:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        ax.cbar = plt.colorbar(sm, ax=ax)
        ax.cbar.set_label('Waiting time at CS')
    else:
        ax.cbar.update_normal(cm.ScalarMappable(norm=norm, cmap=cmap))


def visualize_best_route_animation(route, charging_stations, generations_data, connections, route_points_distances,
                                   queueing_time, interval=500):
    """
    Visualize the best route at each generation using animation.

    Parameters:
    - route: Array of (x, y) coordinates defining the route.
    - charging_stations: Array of (x, y) coordinates of the charging_stations.
    - generations_data: List of best charging stations for each generation.
    - connections: List of tuples defining connections between charging_stations and route segments.
    - route_points_distances: List of distances between consecutive route points.
    - interval: Time interval between frames in milliseconds.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    def update(frame):
        best_charging_stations = generations_data[frame]
        update_plot(ax, route, charging_stations, best_charging_stations, connections, route_points_distances, queueing_time,frame)
        return ax

    ani = FuncAnimation(fig, update, frames=len(generations_data), interval=interval, repeat=False)
    plt.show()


def update_plot_for_dynamic(ax, route, charging_stations, best_charging_stations, connections,
                            queueing_time, distances, starting_point_index, points_to_add):
    ax.clear()

    # Highlight the chosen charging stations.
    # Define custom colormap ranging from green to red
    cmap = plt.cm.get_cmap('RdYlGn_r')  # Reversed RdYlGn colormap

    # Normalize queueing_time for color mapping
    norm = Normalize(vmin=min(queueing_time), vmax=max(queueing_time))

    # Get chosen charging_stations and their queueing_time
    chosen_charging_stations = charging_stations[best_charging_stations]
    chosen_queueing_time = [queueing_time[idx] for idx in best_charging_stations]

    # Scatter chosen charging_stations with color based on queueing_time
    sc = ax.scatter(chosen_charging_stations[:, 0], chosen_charging_stations[:, 1], c=cmap(norm(chosen_queueing_time)), s=100, zorder=5,
                    label='Chosen Charging Stations')

    for i, (x, y) in enumerate(chosen_charging_stations):
        ax.text(x + 0.5, y + 0.5, f'{best_charging_stations[i] + points_to_add }', fontsize=12, color='gold')

    # Plot connections only to the chosen stations
    chosen_connections = [(idx, closest_point(route, charging_stations[idx])) for idx in best_charging_stations]
    for start, end in chosen_connections:
        ax.plot([route[end][0], charging_stations[start][0]], [route[end][1], charging_stations[start][1]], 'r-', linewidth=2)
        mid_x = (route[end][0] + charging_stations[start][0]) / 2
        mid_y = (route[end][1] + charging_stations[start][1]) / 2
        segment_length = np.linalg.norm(route[end] - charging_stations[start])
        ax.text(mid_x, mid_y, f'{segment_length:.2f}', fontsize=10, color='pink')

    # Plot the entire route and annotate all segment lengths
    ax.scatter(route[:, 0], route[:, 1], color='black', alpha=0.5, label='Route Waypoints')
    ax.plot(route[:, 0], route[:, 1], 'green', linestyle='dashed', alpha=0.5)

    for i in range(len(route) - 1):
        mid_x = (route[i, 0] + route[i + 1, 0]) / 2
        mid_y = (route[i, 1] + route[i + 1, 1]) / 2
        ax.text(mid_x, mid_y, f'{distances[i]:.2f}', fontsize=10, color='black')

    # Set the limits of the plot
    plt.xlim(-2, 102)
    plt.ylim(-2, 102)

    ax.set_title(f'Route Visualization - Starting Point Index: {starting_point_index}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True)
    # ax.legend()

    # Add a color bar to indicate penalty values
    if not hasattr(ax, 'cbar') or ax.cbar is None:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        ax.cbar = plt.colorbar(sm, ax=ax)
        ax.cbar.set_label('Waiting time at CS')
    else:
        ax.cbar.update_normal(cm.ScalarMappable(norm=norm, cmap=cmap))


def visualize_all_routes(best_routes):
    fig, ax = plt.subplots(figsize=(10, 8))
    for starting_point_index, (route, charging_stations, best_charging_stations, connections, queueing_time, distances,points_to_add) in enumerate(best_routes):
        update_plot_for_dynamic(ax, route, charging_stations, best_charging_stations,
                                connections, queueing_time, distances, starting_point_index, points_to_add)
        plt.pause(2)  # Pause to display the update (adjust the pause duration as needed)!
    plt.show()
