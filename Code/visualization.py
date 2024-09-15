import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt, image as mpimg
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.widgets import Button
from scipy.spatial.distance import cdist
import matplotlib.cm as cm
from matplotlib.colors import Normalize, to_rgba

from Code.parameters import EV_CAPACITY, POPULATION_SIZE, GENERATIONS, MUTATION_RATE, NUM_POINTS, MAX_STAGNATION, \
    AVERAGE_QUEUEING_TIME, MIN_QUEUEING_TIME, MAX_QUEUEING_TIME


def closest_point(route, charging_station):
    """
    Finds the closest point on the route to the given point.
    """
    distances = cdist([charging_station], route, 'euclidean')
    return np.argmin(distances)


def visualize_clustering(num_clusters, charging_stations, cluster_labels, centroids):
    """
    Visualize the clustered charging_stations
    """
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap('tab10')

    for i in range(num_clusters):
        cluster_charging_stations = charging_stations[cluster_labels == i]
        plt.scatter(cluster_charging_stations[:, 0], cluster_charging_stations[:, 1], color=cmap(i), marker='o',
                    label=f'Cluster {i + 1}')
        plt.scatter(centroids[i, 0], centroids[i, 1], color='black', marker='x', s=100, linewidths=3)

    plt.title('K-Means Clustering of Random Charging Stations')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()


def update_plot(ax, route, charging_stations, best_charging_stations, connections, route_points_distances,
                queueing_time, generation):
    """
    Update the plot with the best route and charging stations at each generation.
    """
    ax.clear()

    # Plot the route as a thin dashed line
    ax.plot(route[:, 0], route[:, 1], color='green', linestyle='--', linewidth=1, alpha=0.8, label='Route')

    # Define custom colormap ranging from green to red
    cmap = plt.cm.get_cmap('RdYlGn_r')  # Reversed RdYlGn colormap

    # Normalize queueing_time for color mapping
    norm = Normalize(vmin=min(queueing_time), vmax=max(queueing_time))

    # Get chosen charging_stations and their queueing_time
    chosen_charging_stations = charging_stations[best_charging_stations]
    chosen_queueing_time = queueing_time[best_charging_stations]

    charging_station_logo = mpimg.imread('Code/assets/cs.png')

    # Plot the charging station logo with corresponding penalty color
    for i, (x, y) in enumerate(chosen_charging_stations):
        color = to_rgba(cmap(norm(chosen_queueing_time[i])))  # Get the RGBA color for current queueing time

        colored_logo = charging_station_logo[:, :, :3].copy()  # Get the RGB channels
        alpha_channel = charging_station_logo[:, :, 3]  # Get the alpha channel

        for c in range(3):  # Adjust each color channel based on the colormap color
            colored_logo[:, :, c] = np.where(alpha_channel > 0, color[c], colored_logo[:, :, c])

        final_logo = np.dstack((colored_logo, alpha_channel))

        imagebox = OffsetImage(final_logo, zoom=0.075, alpha=1)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)

        # Modify the color and add padding to the text
        text_padding_x = -3.0  # Adjust this value to move the text horizontally
        text_padding_y = 2.0  # Adjust this value to move the text vertically
        ax.text(x + text_padding_x, y + text_padding_y, f'{best_charging_stations[i]}', fontsize=10, color='blue',
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))

    # Plot connections only to the chosen stations
    chosen_connections = [(idx, closest_point(route, charging_stations[idx])) for idx in best_charging_stations]
    for start, end in chosen_connections:
        ax.plot([route[end][0], charging_stations[start][0]], [route[end][1], charging_stations[start][1]],
                color='#FF5962', linewidth=2)
        mid_x = (route[end][0] + charging_stations[start][0]) / 2
        mid_y = (route[end][1] + charging_stations[start][1]) / 2
        segment_length = np.linalg.norm(route[end] - charging_stations[start])
        ax.text(mid_x, mid_y, f'{segment_length:.2f}', fontsize=10, color='#221BDC')

    # Plot the entire route and annotate all segment lengths
    ax.scatter(route[:, 0], route[:, 1], color='black', alpha=0.5, label='Route Waypoints')

    for i in range(len(route) - 1):
        mid_x = (route[i, 0] + route[i + 1, 0]) / 2
        mid_y = (route[i, 1] + route[i + 1, 1]) / 2
        ax.text(mid_x, mid_y, f'{route_points_distances[i]:.2f}', fontsize=10, color='black')

    # Adjust generation text
    ax.text(0.75, 0.95, f'Generation: {generation}', transform=ax.transAxes, fontsize=14,
            horizontalalignment='center', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)

    ax.set_title("Chosen Route Visualization")
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True)


def visualize_best_route_animation(route, charging_stations, generations_data, connections, route_points_distances,
                                   queueing_time, interval=500):
    fig, ax = plt.subplots(figsize=(12, 8))
    show_all_points = False

    def update(frame):
        best_charging_stations = generations_data[frame]
        update_plot(ax, route, charging_stations, best_charging_stations, connections, route_points_distances,
                    queueing_time, frame)
        return ax

    ani = FuncAnimation(fig, update, frames=len(generations_data), interval=interval, repeat=False)
    plt.show()


def update_plot_for_dynamic(ax, route, charging_stations, best_charging_stations, connections,
                            queueing_time, distances, starting_point_index, points_to_add, subset_cluster_labels,
                            subset_centroids, starting_point_cluster, final_chromosome, show_all_points=False):
    ax.clear()
    cmap = plt.cm.get_cmap('RdYlGn_r')  # Colormap to represent queueing times
    norm = Normalize(vmin=MIN_QUEUEING_TIME, vmax=MAX_QUEUEING_TIME)

    if show_all_points:
        for idx, (x, y) in enumerate(charging_stations):
            if idx in best_charging_stations:
                continue  # Skip already selected stations

            if subset_cluster_labels[idx] == starting_point_cluster:
                # Set color based on the actual queueing time for stations in the current cluster
                color = cmap(norm(queueing_time[idx]))
            else:
                # Use the average queueing time for stations not in the current cluster
                color = cmap(norm(AVERAGE_QUEUEING_TIME))

                # Plot the charging station
            ax.scatter(x, y, color=color, alpha=0.7)
            ax.text(x, y, f'{idx + points_to_add}', fontsize=8, color='gray')

    # Highlight the chosen charging stations
    chosen_charging_stations = charging_stations[best_charging_stations]
    chosen_queueing_time = [queueing_time[idx] for idx in best_charging_stations]

    charging_station_logo = mpimg.imread('Code/assets/cs.png')

    # Plot the charging station logo with corresponding penalty color
    for i, (x, y) in enumerate(chosen_charging_stations):
        if subset_cluster_labels[
            best_charging_stations[i]] == starting_point_cluster:  # Check if the station is in the current cluster
            color = to_rgba(cmap(norm(chosen_queueing_time[i])))  # Get the RGBA color for current queueing time
        else:
            color = to_rgba(cmap(norm(AVERAGE_QUEUEING_TIME)))  # Use the fixed color for stations not in the current cluster

        colored_logo = charging_station_logo[:, :, :3].copy()  # Get the RGB channels
        alpha_channel = charging_station_logo[:, :, 3]  # Get the alpha channel

        for c in range(3):  # Adjust each color channel based on the colormap color
            colored_logo[:, :, c] = np.where(alpha_channel > 0, color[c], colored_logo[:, :, c])

        final_logo = np.dstack((colored_logo, alpha_channel))

        imagebox = OffsetImage(final_logo, zoom=0.075, alpha=1)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)

        # Modify the color and add padding to the text
        text_padding_x = -3.0  # Adjust this value to move the text horizontally
        text_padding_y = 2.0  # Adjust this value to move the text vertically
        ax.text(x + text_padding_x, y + text_padding_y, f'{best_charging_stations[i] + points_to_add}',
                fontsize=10, color='blue', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))

    # Plot connections only to the chosen stations
    chosen_connections = [(idx, closest_point(route, charging_stations[idx])) for idx in best_charging_stations]
    for start, end in chosen_connections:
        ax.plot([route[end][0], charging_stations[start][0]], [route[end][1], charging_stations[start][1]],
                color='#FF5962',
                linewidth=2)
        mid_x = (route[end][0] + charging_stations[start][0]) / 2
        mid_y = (route[end][1] + charging_stations[start][1]) / 2
        segment_length = np.linalg.norm(route[end] - charging_stations[start])
        ax.text(mid_x, mid_y, f'{segment_length:.2f}', fontsize=10, color='#221BDC')

    # Rotate car icon at the starting point to align with the direction of the route
    car_icon = mpimg.imread('Code/assets/car_icon.png')
    gold_color = [102 / 255, 153 / 255, 255 / 255]  # RGB for gold

    if len(route) > 1:
        start_x, start_y = route[0]
        next_x, next_y = route[1]

        angle = np.degrees(np.arctan2(next_y - start_y, next_x - start_x))

        rotated_car_icon = ndimage.rotate(car_icon, angle, reshape=True)

        imagebox = OffsetImage(rotated_car_icon, zoom=0.06, alpha=1)

        ab = AnnotationBbox(imagebox, (start_x, start_y), frameon=False)
        ax.add_artist(ab)

        updated_best_charging_stations = []
        for i in range(len(best_charging_stations)):
            updated_best_charging_stations.append(best_charging_stations[i] + points_to_add)  # Add to each element
        if updated_best_charging_stations and updated_best_charging_stations[0] in final_chromosome and (
                updated_best_charging_stations[0], 0) in connections:
            # Apply the color to the car icon (assuming the car icon has an alpha channel)
            colored_car_icon = car_icon[:, :, :3].copy()
            alpha_channel = car_icon[:, :, 3]  # Get the alpha channel
            # Adjust each color channel based on the chosen color
            for c in range(3):
                colored_car_icon[:, :, c] = np.where(alpha_channel > 0, gold_color[c], colored_car_icon[:, :, c])
            # Combine the colored car icon with the alpha channel
            final_car_icon = np.dstack((colored_car_icon, alpha_channel))
            # Display the colored car icon
            rotated_car_icon = ndimage.rotate(final_car_icon, angle, reshape=True)
            imagebox = OffsetImage(rotated_car_icon, zoom=0.06, alpha=1)
            ab = AnnotationBbox(imagebox, (start_x, start_y), frameon=False)
            ax.add_artist(ab)

    # Plot the entire route and annotate all segment lengths
    ax.scatter(route[:, 0], route[:, 1], color='black', alpha=0.5, label='Route Waypoints')
    ax.plot(route[:, 0], route[:, 1], color='#39A155', linestyle='dashed', alpha=0.5)

    for i in range(len(route) - 1):
        mid_x = (route[i, 0] + route[i + 1, 0]) / 2
        mid_y = (route[i, 1] + route[i + 1, 1]) / 2
        ax.text(mid_x, mid_y, f'{distances[i]:.2f}', fontsize=10, color='black')

    # Set the limits of the plot
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)

    ax.set_title(f'Route Visualization - Starting Point Index: {starting_point_index}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True)

    # Add a color bar to indicate queueing times
    if not hasattr(ax, 'cbar') or ax.cbar is None:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Queueing Time at CS')
        ax.cbar = cbar
    else:
        ax.cbar.update_normal(cm.ScalarMappable(norm=norm, cmap=cmap))

    # Parameters to visualize
    params = [
        f'EV Capacity: {EV_CAPACITY}',
        f'Population Size: {POPULATION_SIZE}',
        f'Generations: {GENERATIONS}',
        f'Mutation Rate: {MUTATION_RATE}',
        f'Num Points: {NUM_POINTS}',
        f'Average Queueing Time: {AVERAGE_QUEUEING_TIME}',
        f'Max Stagnation: {MAX_STAGNATION}'
    ]

    # Add text above the title for parameters
    param_text = ' | '.join(params)
    ax.text(0.5, 1.10, param_text, transform=ax.transAxes, horizontalalignment='center',
            fontsize=8, color='black', bbox=dict(facecolor='#E1E1E1', edgecolor='grey', boxstyle='round,pad=0.3'))


# Function to visualize all routes and add button control
def visualize_all_routes(best_routes, cluster_labels, centroids, starting_point_clusters, final_chromosome):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Initialize index for starting point
    current_index = 0
    show_all_points = False

    def update_plot_with_toggle():
        """Helper function to update the plot based on the current state."""
        route, charging_stations, best_charging_stations, connections, queueing_time, distances, points_to_add = \
            best_routes[current_index]
        subset_cluster_labels = cluster_labels[points_to_add:]
        subset_centroids = centroids
        starting_point_cluster = starting_point_clusters[current_index]
        update_plot_for_dynamic(ax, route, charging_stations, best_charging_stations,
                                connections, queueing_time, distances, current_index, points_to_add,
                                subset_cluster_labels, subset_centroids, starting_point_cluster, final_chromosome,
                                show_all_points=show_all_points)
        plt.draw()

    def next_plot(event):
        """Function to move to the next plot."""
        nonlocal current_index
        current_index = (current_index + 1) % len(best_routes)
        update_plot_with_toggle()

    def toggle_display(event):
        """Function to toggle the display of all points."""
        nonlocal show_all_points
        show_all_points = not show_all_points
        update_plot_with_toggle()

    # Initial plot
    update_plot_with_toggle()
    # Create a button and set its position
    ax_button_next = plt.axes([0.78, 0.01, 0.08, 0.04])
    button_next = Button(ax_button_next, label='Next', color='#2A75A9', hovercolor='#5079C1')
    button_next.on_clicked(next_plot)

    ax_button_toggle = plt.axes([0.64, 0.01, 0.12, 0.04])
    button_toggle = Button(ax_button_toggle, label='Toggle Points', color='#4C8C4A', hovercolor='#72A77E')
    button_toggle.on_clicked(toggle_display)

    plt.show()

