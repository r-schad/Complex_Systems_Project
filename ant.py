import random

import numpy as np

from network import LatticeNetwork

class Ant:
    def __init__(self, vec: np.ndarray, pos: tuple, alpha: float, beta: float, delta: float):
        # initialize ant with the document vector
        self.vec = vec
        self.pos = pos

        # initialize hyperparams
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def decide_next_position(self, network: LatticeNetwork, q: float = 0.2) -> bool:
        neighbors = network.get_neighbors(*self.pos)
        centroid_vecs = [network.get_centroid_pheromone_vec(r, c) for r, c in neighbors]
        pheromone_vecs = [network.get_pheromone_vec(r, c) for r, c in neighbors]
        edge_pheromones = [self.find_edge_pheromone(centroid_vecs[i], pheromone_vecs[i]) for i in range(len(neighbors))]
        pheromones = [self.pheromone_weighting(sigma) for sigma in edge_pheromones]
        stopped = False
        if random.random() < q:
            i = np.argmax(pheromones)
            current_pheromone = self.find_edge_pheromone(network.get_centroid_pheromone_vec(*self.pos), 
                                                         network.get_pheromone_vec(*self.pos))
            current_pheromone = self.pheromone_weighting(current_pheromone)
            if pheromones[i] < current_pheromone:
                stopped = True
        else:
            probs = np.array(pheromones) / np.sum(pheromones)
            i = int(np.random.choice(np.arange(len(neighbors)), 1, replace=False, p=probs))
        if not stopped:
            new_pos = neighbors[i]
            # TODO: Implement previous move tracking
            self.pos = new_pos
        return stopped

    # do a roulette wheel decision with weighted probabilities
    def roulette_wheel(self, probs: np.ndarray, num_samples: int = 1) -> float:
        return np.random.choice(np.arange(len(probs)), num_samples, p=probs)

    # apply ant pheromone weighting
    def pheromone_weighting(self, sigma: float) -> float:
        # TODO: are we doing the ant path smoothing factor here? See equation (9) in Fernandes et al.
        return (1 + (self.delta / (1 + (sigma * self.delta)))) ** self.beta

    # compute the pheromone of the edge between two nodes (equation (10) in Fernandes et al.)
    def find_edge_pheromone(self, centroid_pheromone: np.ndarray, node_pheromone: np.ndarray) -> float:
        unweight_pheromone = np.linalg.norm(centroid_pheromone - node_pheromone)
        return self.pheromone_weighting(unweight_pheromone)

    # get pheromone update reinforce (R in equation (12) in Fernandes et al.)
    def get_update_reinforce(self, centroid_pheromone: np.ndarray, node_pheromone: np.ndarray) -> float:
        return self.alpha * (1 - (np.linalg.norm(centroid_pheromone - node_pheromone) / len(node_pheromone)))

    # update the pheromone vector for the specified node (equation (11) in Fernandes et al.)
    def get_new_pheromone_vec(self, network: LatticeNetwork) -> np.ndarray:
        centroid_pheromone = network.get_centroid_pheromone_vec(*self.pos)
        node_pheromone = network.get_pheromone_vec(*self.pos)
        reinforce = self.get_update_reinforce(centroid_pheromone, node_pheromone)
        return node_pheromone + reinforce * (self.vec - node_pheromone)