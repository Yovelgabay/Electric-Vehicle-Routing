
# Electric Vehicle Routing Optimization

This project aims to optimize electric vehicle (EV) routes using K-Means clustering and a Genetic Algorithm (GA). It addresses the real-world challenge of determining the most efficient route for an EV from point A to point B, considering stops at charging stations while accounting for dynamic factors such as queuing delays and travel distance.

## Project Overview

The system dynamically generates routes for electric vehicles, incorporating real-world charging constraints such as battery capacity, travel distance, and queuing times at charging stations. The project uses a two-phase approach:

1. **K-Means Clustering**: Partition charging stations based on geographical proximity, allowing for more manageable routing based on the current cluster of the EV.
2. **Genetic Algorithm Optimization**: The system dynamically optimizes the total travel distance and stops by running a genetic algorithm that selects the best charging stations based on distance and waiting times.

## Features

- **Dynamic Route Optimization**: The route is recalculated dynamically based on the EV's current location, remaining battery, and real-time updates at each step.
- **Flexible Algorithm**: Users can select different combinations of crossover, mutation, and selection methods within the Genetic Algorithm, allowing for experimentation with various optimization techniques.
- **Realistic Charging Constraints**: The system integrates queuing times at charging stations, creating more realistic scenarios for EV routing.
- **Visualization**: The system provides visualizations of routes, charging stations, and dynamically evolving solutions through graphs and animations.

## Getting Started

### Prerequisites

- **Python 3.7 or later**
- **Required Python Packages**:
  - numpy
  - matplotlib
  - scikit-learn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Yovelgabay/Electric-Vehicle-Routing.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Electric-Vehicle-Routing
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

To run the main optimization script:

```bash
python main.py
```

### Customizing Parameters

The system allows users to adjust the parameters in `parameters.py` to suit various routing scenarios. Here are a few key parameters you can customize:

- `EV_CAPACITY`: The total battery capacity of the EV.
- `NUM_ROUTE_POINTS`: The number of route points along the EV's journey.
- `POPULATION_SIZE`: The number of candidate solutions (routes) used in the Genetic Algorithm.
- `GENERATIONS`: The number of generations for which the GA evolves the population.
- `MUTATION_RATE`: The probability of mutation during the GA process.
- `NUM_POINTS`: The number of charging stations available along the route.
- `CHECK_POINTS`: The list of predefined route checkpoints the EV must pass through.

### Visualizations

The system provides visualizations of the optimal routes and the charging station points. Dynamic animations of the best route as it evolves over time are also available.


## Contributing

Feel free to submit issues or pull requests to improve the project. Contributions to enhance the algorithm, introduce new features, or fix bugs are always welcome.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This project was supervised by Prof. Miri Weiss-Cohen and developed as part of the Capstone Project at ORT Braude College.
