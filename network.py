import numpy as np

class Lattice_Network():
    def __init__(self, network_shape: tuple, embedding_dimension: int):
        if type(network_shape) != tuple:
            raise ValueError('network_shape should be a numpy shape (i.e. a tuple)')
    
        if type(embedding_dimension) != int:
            raise ValueError('embedding_dimension should be an int')

        # initialize pheromone vectors
        unnormalized_pheromones = np.random.rand(network_shape[0], network_shape[1], embedding_dimension) 
        # normalize each node's pheromone vector so it sums to 1.0 
        self.pheromones = unnormalized_pheromones / np.sum(unnormalized_pheromones, axis=2, keepdims=2)

        # initialize 8-neighbor neighborhood adjacency list
        self.neighbors = np.ndarray(network_shape, dtype=list)
        for row in range(network_shape[0]):
            for col in range(network_shape[1]):
                self.neighbors[row][col] = []

                if row != 0:
                    north = (row - 1, col)
                    self.neighbors[row][col].append(north)

                    if col != 0:
                        northwest = (row - 1, col - 1)
                        self.neighbors[row][col].append(northwest)

                    if col != network_shape[1] - 1:
                        northeast = (row - 1, col + 1)
                        self.neighbors[row][col].append(northeast)
                
                if row != network_shape[0] - 1:
                    south = (row + 1, col)
                    self.neighbors[row][col].append(south)

                    if col != 0:
                        southwest = (row + 1, col - 1)
                        self.neighbors[row][col].append(southwest)

                    if col != network_shape[1] - 1:
                        southeast = (row + 1, col + 1)
                        self.neighbors[row][col].append(southeast)

                if col != 0:
                    west = (row, col - 1)
                    self.neighbors[row][col].append(west)

                if col != network_shape[1] - 1:
                    east = (row, col + 1)
                    self.neighbors[row][col].append(east)\
                    