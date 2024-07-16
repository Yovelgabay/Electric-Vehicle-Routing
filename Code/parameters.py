import numpy as np

EV_CAPACITY = 100
NUM_ROUTE_POINTS = 25
POPULATION_SIZE = 2000
GENERATIONS = 100
MUTATION_RATE = 0.2
NUM_POINTS = 500
MAX_STAGNATION = 10
AVERAGE_WAITING_TIME = 10

CHECK_POINTS = [
    np.array([10, 90]),
    np.array([30, 5]),
    np.array([60, 90]),
    np.array([90, 40])
]
