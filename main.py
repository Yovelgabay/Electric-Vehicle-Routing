import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from GA import genetic_algorithm
from kmeans import kmeans_clustering


def generate_zigzag_route(num_points, scale, turn_amplitude, seed):
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


def generate_random_points_and_penalties(seed, num_points, scale, route):
    np.random.seed(seed)
    points = np.random.rand(num_points, 2) * scale
    np.random.seed(seed + 1)
    penalties = np.random.uniform(1, 20, num_points)
    intersections = get_intersection_points(route, points)
    station_ids = sort_stations_by_route(intersections)
    points_with_ids = [(i, points[i]) for i in station_ids]
    return points_with_ids, penalties


def visualize_route(points, route, title, penalties, connections=[]):
    plt.figure(figsize=(10, 8))
    plt.plot(route[:, 0], route[:, 1], 'r-o', label='Route')
    plt.scatter(points[:, 0], points[:, 1], color='blue', label='Charging Stations')
    for i, (x, y) in enumerate(points):
        plt.text(x + 0.5, y + 0.5, f'{i}', fontsize=12, color='green')
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
    distances = cdist([point], route, 'euclidean')
    return np.argmin(distances)


def get_intersection_points(route, points):
    intersections = []
    for idx, point in enumerate(points):
        closest_idx = closest_point(route, point)
        intersections.append((idx, point, closest_idx))
    return intersections


def sort_stations_by_route(intersections):
    intersections.sort(key=lambda x: x[2])  # Sort based on closest route point index
    return [x[0] for x in intersections]  # Return original indices of points


# Generating data
num_route_points = 20
route = generate_zigzag_route(num_route_points, 100, 10, seed=53)
points_matrix, penalties = generate_random_points_and_penalties(2, 20, 100, route)

# Extract points from points_matrix
points = np.array([point[1] for point in points_matrix])

# Visualize initial setup
intersections = get_intersection_points(route, points)
connections = [(idx, closest_point(route, pt)) for idx, pt, _ in intersections]
visualize_route(points, route, 'Zigzag Route Visualization', penalties, connections)

# Apply clustering and genetic algorithm
kmeans_clustering(points)
best_route_indices = genetic_algorithm(points_matrix, route, connections, population_size=20, generations=50, mutation_rate=0.1,
                                       penalties=penalties)

# Find points corresponding to the best route
best_route_points = [points_matrix[i][1] for i in best_route_indices]
print("Best Route:", best_route_indices)
# Find connections for the optimized route
optimized_connections = [(idx, closest_point(route, pt)) for idx, pt in enumerate(best_route_points)]
optimized_route = [route[0]]
for start, end in optimized_connections:
    optimized_route.extend([route[end], points[start], route[end]])
optimized_route.append(route[-1])
optimized_route = np.array(optimized_route)

# Visualize the optimized route
visualize_route(points, optimized_route, 'Optimized Route Visualization', penalties, optimized_connections)
