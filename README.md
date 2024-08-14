# Enhancing Electric Vehicle Routing Efficiency Through K-Means Clustering and Genetic Algorithm Optimization

---

## Introduction

This project focuses on optimizing electric vehicle (EV) routing by combining K-means clustering and a genetic algorithm. The goal is to enhance routing efficiency by dynamically evaluating routes and incorporating real-time data.

---

## Clustering

Initially, we aimed to simplify the problem by clustering charging stations. Each route point was assigned to the nearest cluster of charging stations, with the genetic algorithm evaluating optimal stops. Charging stations within the same cluster as the current route point were assumed to have real queueing times, while those outside the cluster had average waiting times. This method enhances model realism by using actual data when route points are close to charging stations.

We originally planned to determine the optimal number of clusters K using the silhouette score. However, we found that clustering did not significantly reduce complexity and that the silhouette score suggested too few clusters. Therefore, we adjusted K to be the number of route points divided by 3, which demonstrated the dynamic features of our system while maintaining manageable complexity.

---

## Route Representation

The route is represented as a sequence of points with (x, y) coordinates. These points include a starting point, several checkpoints, and an ending point. The points act as exit points to the charging stations, allowing vehicles to leave the main route to access charging stations and then return.

---

## Charging Stations

Charging stations are represented as random points with associated queuing time penalties ranging from 0 to 20 minutes. We calculate the Euclidean distance between each station and every route point to determine the closest route point, which serves as the exit point to the station. This mapping ensures:

1. A charging station can only be accessed via its corresponding route point.
2. Each route point can lead to multiple charging stations, which are sorted based on proximity for the genetic algorithm.

---

## Genetic Algorithm for Routing Problem

### Overview

The genetic algorithm (GA) is used to generate high-quality solutions for optimization problems. It starts with an initial population of chromosomes and evolves through selection, crossover, and mutation to find optimal routes.

### Genetic Representation

Each chromosome represents a potential solution, consisting of sequences of positive integers where each integer is the ID of a charging station. Chromosomes vary in length, allowing the algorithm to explore different routing paths.

### Crossover

Crossover combines genetic information from two parent chromosomes to produce offspring:

- **Crossover with Common Nodes:** When parents share common nodes, crossover occurs at a random common node, swapping segments between parents.
- **Crossover with No Common Nodes:** Crossover occurs at a midpoint, combining segments from both parents to create offspring.

Offspring are sorted to ensure valid routes.

### Mutation

Mutation introduces genetic diversity by making random changes to chromosomes. The mutation process includes:

1. **Random Locus Selection:** Select a random position in the chromosome.
2. **Determine Valid Range:** Ensure the new gene is within a valid range to maintain route feasibility.
3. **Replace Gene:** Replace the old gene with a new one within the valid range.

Mutation includes:
- Replacing a single station
- Adding a new station
- Removing a single station

### Population Initialization

The initial population is generated randomly, including:
1. Shortest Valid Path
2. Longest Path
3. Random Paths

This diversity is essential for effective exploration of solutions.

### Fitness Function

The fitness function evaluates how well a route minimizes distance and waiting time while adhering to EV capacity constraints. It calculates penalties for invalid routes, total distance, and waiting times, ensuring efficient routing.

### Selection

**Tournament Selection** is used to choose parents for crossover. A subset of chromosomes is randomly selected, and the one with the highest fitness is chosen as a parent.

### Stagnation

Stagnation indicates minimal improvement over generations. The algorithm terminates if no significant improvement is observed for a set number of generations.

### Dynamic Process

The algorithm dynamically updates the route based on the EV’s current position, recalculating the optimal route and updating charging station information accordingly.

---

## Research Process

We evaluated various algorithms for:
- Selection operator
- Crossover opreater
- Mutation operator
