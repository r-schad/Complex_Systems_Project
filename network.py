import numpy as np

class Lattice_Network():
    def __init__(self, network_shape):
        if type(network_shape) != tuple:
            raise ValueError('network_shape should be a numpy shape (i.e. a tuple)')
    
        self.grid = np.ndarray(network_shape, dtype=list)
        for row in range(network_shape[0]):
            for col in range(network_shape[1]):
                self.grid[row][col] = []

                if row != 0:
                    north = (row - 1, col)
                    self.grid[row][col].append(north)

                    if col != 0:
                        northwest = (row - 1, col - 1)
                        self.grid[row][col].append(northwest)

                    if col != network_shape[1] - 1:
                        northeast = (row - 1, col + 1)
                        self.grid[row][col].append(northeast)
                
                if row != network_shape[0] - 1:
                    south = (row + 1, col)
                    self.grid[row][col].append(south)

                    if col != 0:
                        southwest = (row + 1, col - 1)
                        self.grid[row][col].append(southwest)

                    if col != network_shape[1] - 1:
                        southeast = (row + 1, col + 1)
                        self.grid[row][col].append(southeast)

                if col != 0:
                    west = (row, col - 1)
                    self.grid[row][col].append(west)

                if col != network_shape[1] - 1:
                    east = (row, col + 1)
                    self.grid[row][col].append(east)

        
net = Lattice_Network((10,10))
print(net.grid)