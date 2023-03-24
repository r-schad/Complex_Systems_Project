import numpy as np

from network import Lattice_Network

class Ant:
    def __init__(self, vec: np.ndarray, pos: tuple, alpha: float, beta: float, delta: float):
        # initialize ant with the document vector
        self.vec = vec
        self.pos = pos

        # initialize hyperparams
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def decide_next_position(self, network: Lattice_Network):
        # TODO: Implement this
        pass

    # do a roulette wheel decision with weighted probabilities
    def roulette_wheel(self, probs: np.ndarray, num_samples: int = 1) -> float:
        return np.random.choice(np.arange(len(probs)), num_samples, p=probs)

    # apply ant pheremone weighting
    def pheremone_weighting(self, sigma: float) -> float:
        # TODO: are we doing the ant path smoothing factor here? See equation (9) in Fernandes et al.
        return (1 + (self.delta / (1 + (sigma * self.delta)))) ** self.beta

    # compute the pheremone of the edge between two nodes (equation (10) in Fernandes et al.)
    def find_edge_pheremone(self, centroid_pheremone: np.ndarray, node_pheremone: np.ndarray) -> float:
        unweight_pheremone = np.linalg.norm(centroid_pheremone - node_pheremone)
        return self.pheremone_weighting(unweight_pheremone)

    # get pheremone update reinforce (R in equation (12) in Fernandes et al.)
    def get_update_reinforce(self, centroid_pheremone: np.ndarray, node_pheremone: np.ndarray) -> float:
        return self.alpha * (1 - (np.linalg.norm(centroid_pheremone - node_pheremone) / len(node_pheremone)))

    # update the pheremone vector for the specified node (equation (11) in Fernandes et al.)
    def get_new_pheremone_vec(self, centroid_pheremone: np.ndarray, node_pheremone: np.ndarray) -> np.ndarray:
        reinforce = self.get_update_reinforce(centroid_pheremone, node_pheremone)
        return node_pheremone + reinforce * (self.vec - node_pheremone)