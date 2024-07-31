import numpy as np

EV_CAPACITY = 100
NUM_ROUTE_POINTS = 30
POPULATION_SIZE = 5000
GENERATIONS = 100
MUTATION_RATE = 1
NUM_POINTS = 100
MAX_STAGNATION = 15
AVERAGE_QUEUEING_TIME = 12

CHECK_POINTS = [
    np.array([5, 90]),
    np.array([12, 30]),
    np.array([20, 5]),
    # np.array([26, 40]),
    np.array([36, 80]),
    np.array([40, 90]),
    # np.array([50, 60]),
    np.array([70, 10]),
    np.array([90, 50])
]
