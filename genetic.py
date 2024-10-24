
import time
import numpy as np
import random
import sys
INF = sys.maxsize


def adjMatFromFile(filename):
    """ Create an adj/weight matrix from a file with verts, neighbors, and weights. """
    f = open(filename, "r")
    n_verts = int(f.readline())
    print(f" n_verts = {n_verts}, please wait for algorithm to finish...")
    adjmat = [[None] * n_verts for i in range(n_verts)]
    for i in range(n_verts):
        adjmat[i][i] = 0
    for line in f:
        int_list = [int(i) for i in line.split()]
        vert = int_list.pop(0)
        assert len(int_list) % 2 == 0
        n_neighbors = len(int_list) // 2
        neighbors = [int_list[n] for n in range(0, len(int_list), 2)]
        distances = [int_list[d] for d in range(1, len(int_list), 2)]
        for i in range(n_neighbors):
            adjmat[vert][neighbors[i]] = distances[i]
    f.close()
    return adjmat


def geneticAlgo(g, max_generations=4000, population_size=600, base_mutation_rate=0.075, explore_rate=0.3):
    """
    Implements a genetic algorithm to find the shortest path in a graph with dynamic mutation adjustment.
    
    Parameters:
        g (list): Adjacency matrix representing the graph.
        max_generations (int): Maximum number of generations to run the algorithm.
        population_size (int): Size of the population in each generation.
        base_mutation_rate (float): Base rate at which mutations occur.
        explore_rate (float): Rate at which exploration of new paths occurs (not currently used).
        
    Returns:
        dict: A dictionary containing the best path found, its distance, the average distances per generation, and the final mutation rate.
    """
    mutation_rate = base_mutation_rate
    last_best_distance = float('inf')  # Assuming INF was meant to be float('inf')
    stagnation_count = 0

    def create_random_path(num_cities):
        """ Creates a random path through all cities. """
        path = list(range(num_cities))
        random.shuffle(path)
        return path

    def path_distance(path):
        """ Calculates the total distance of a given path. """
        return sum(g[path[i]][path[i+1]] for i in range(len(path) - 1)) + g[path[-1]][path[0]]

    def mutate(path, mutation_rate):
        """ Mutates a path by swapping two cities if a random threshold is met. """
        if random.random() < mutation_rate:
            swap_idx1, swap_idx2 = random.sample(range(len(path)), 2)
            path[swap_idx1], path[swap_idx2] = path[swap_idx2], path[swap_idx1]
        return path

    def ordered_crossover(parent1, parent2):
        """ Performs an ordered crossover between two parent paths. """
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = parent1[start:end]
        fill_items = [item for item in parent2 if item not in child[start:end]]
        fill_pos = list(filter(lambda x: child[x] is None, range(size)))
        for idx, item in zip(fill_pos, fill_items):
            child[idx] = item
        return child

    def select(population, scores):
        """ Selects a fraction of the population to reproduce based on scores. """
        selected_indices = np.argsort(scores)[:int(len(scores) * explore_rate)]
        return [population[i] for i in selected_indices]

    population = [create_random_path(len(g)) for _ in range(population_size)]
    best_distance = float('inf')
    best_path = None
    avg_path_dist_each_generation = []

    for generation in range(max_generations):
        distances = [path_distance(path) for path in population]
        avg_distance = np.mean(distances)
        avg_path_dist_each_generation.append(avg_distance)

        current_best_distance = min(distances)
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_path = population[distances.index(best_distance)]
            stagnation_count = 0  # reset stagnation count on improvement
        else:
            stagnation_count += 1
        
        # Adapt mutation rate based on stagnation
        if stagnation_count > 50:  # Stagnation threshold
            mutation_rate *= 1.05
        else:
            mutation_rate *= 0.95

        selected = select(population, distances)
        new_population = [best_path.copy()]  # Include the best path from the current generation
        while len(new_population) < population_size:
            parents = random.sample(selected, 2)
            child1 = ordered_crossover(parents[0], parents[1])
            child2 = ordered_crossover(parents[1], parents[0])
            new_population.append(mutate(child1, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(mutate(child2, mutation_rate))

        population = new_population

    return {
        'path': best_path + [best_path[0]],  # Ensure the path is circular
        'path_distance': best_distance,
        'generation_distances': avg_path_dist_each_generation,
        'final_mutation': mutation_rate
    }


if __name__ == '__main__':
    """ Load the graph """
    g = adjMatFromFile("n100.txt")

    start_time = time.time()
    res_ga = geneticAlgo(g)
    elapsed_time_ga = time.time() - start_time
    print(f"GenAlgo runtime: {elapsed_time_ga:.2f}")
    print(f"  path: {res_ga['path']}")
    print(f"  path dist: {res_ga['path_distance']}")
    print(f"  final mutation rate: {res_ga['final_mutation']}")

