import numpy as np

from typing import List, Tuple

class Lattice_Network():
    def __init__(self, network_shape: Tuple, embedding_dimension: int, evap_factor: float, 
                 centroid_radius: int = 1):
        # initialize centroid radius and evaporation factor
        self.centroid_radius = centroid_radius
        self.evap_factor = evap_factor

        if type(network_shape) != tuple:
            raise ValueError('network_shape should be a numpy shape (i.e. a tuple)')
    
        if type(embedding_dimension) != int:
            raise ValueError('embedding_dimension should be an int')

        # initialize pheromone vectors
        unnormalized_pheromones = np.random.rand(network_shape[0], network_shape[1], embedding_dimension) 
        # normalize each node's pheromone vector to an L2-norm of 1
        self.pheromones = unnormalized_pheromones / np.linalg.norm(unnormalized_pheromones, axis=2, keepdims=True)
        self.init_pheremones = np.copy(self.pheremones)

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
                    self.neighbors[row][col].append(east)
    
    def get_pheremone_vec(self, row: int, col: int) -> np.ndarray:
        return self.pheremones[row][col]

    def get_centroid_pheremone_vec(self, row: int, col: int) -> np.ndarray:
        # get max row and column of the pheremone map
        r, c, _ = self.pheromones.shape
        min_row, max_row = max(0, row - self.centroid_radius), min(r - 1, row + self.centroid_radius)
        min_col, max_col = max(0, col - self.centroid_radius), min(c - 1, col + self.centroid_radius)
        # extract all pheremones in the target region
        region_pheremones = self.pheremones[min_row:max_row][min_col:max_col]
        return np.mean(region_pheremones, axis=[0, 1])
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple]:
        return self.neighbors[row][col]

    def evaporate_pheremones(self):
        self.pheremones -= self.evap_factor * self.init_pheremones