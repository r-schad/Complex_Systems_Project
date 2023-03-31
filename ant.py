import numpy as np
import matplotlib.pyplot as plt

from network import LatticeNetwork

rng = np.random.default_rng(seed=0)

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
        if rng.uniform() < q:
            i = np.argmax(pheromones)
            current_pheromone = self.find_edge_pheromone(network.get_centroid_pheromone_vec(*self.pos), 
                                                         network.get_pheromone_vec(*self.pos))
            current_pheromone = self.pheromone_weighting(current_pheromone)
            if pheromones[i] < current_pheromone:
                stopped = True
        else:
            probs = np.array(pheromones) / np.sum(pheromones)
            i = int(rng.choice(np.arange(len(neighbors)), 1, replace=False, p=probs))
        if not stopped:
            new_pos = neighbors[i]
            # TODO: Implement previous move tracking
            self.pos = new_pos
        return stopped

    # do a roulette wheel decision with weighted probabilities
    def roulette_wheel(self, probs: np.ndarray, num_samples: int = 1) -> float:
        return rng.choice(np.arange(len(probs)), num_samples, p=probs)

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
        # TODO: do we need to divide by the number of variables in the node pheromone if all our vectors are normalized?
        return self.alpha * (1 - (np.linalg.norm(centroid_pheromone - node_pheromone) / len(node_pheromone)))

    # update the pheromone vector for the specified node (equation (11) in Fernandes et al.)
    def get_new_pheromone_vec(self, network: LatticeNetwork) -> np.ndarray:
        centroid_pheromone = network.get_centroid_pheromone_vec(*self.pos)
        node_pheromone = network.get_pheromone_vec(*self.pos)
        reinforce = self.get_update_reinforce(centroid_pheromone, node_pheromone)
        return node_pheromone + reinforce * (self.vec - node_pheromone)

if __name__ == "__main__":
    network = LatticeNetwork((5, 5), 3, 0.99, rng=rng)
    ant1 = Ant(np.array([1, 0, 0]), (0, 0), 1, 1, 1)
    ant2 = Ant(np.array([0, 1, 0]), (2, 2), 1, 1, 1)
    ant3 = Ant(np.array([0, 0, 1]), (4, 4), 1, 1, 1)
    status1, status2, status3 = False, False, False

    i = 0
    while not (status1 or status2 or status3) and i < 10000:
        network.deposit_pheromone(ant1.vec, *ant1.pos)
        network.deposit_pheromone(ant2.vec, *ant2.pos)
        network.deposit_pheromone(ant3.vec, *ant3.pos)
        ant1.decide_next_position(network, 0.5)
        ant2.decide_next_position(network, 0.5)
        ant3.decide_next_position(network, 0.5)
        network.evaporate_pheromones()
        i += 1

    diff1 = np.linalg.norm(network.pheromones - ant1.vec, axis=-1)
    diff2 = np.linalg.norm(network.pheromones - ant2.vec, axis=-1)
    diff3 = np.linalg.norm(network.pheromones - ant3.vec, axis=-1)
    print(f"Status 1: {status1}, Pos: {ant1.pos}")
    print(f"Status 2: {status2}, Pos: {ant2.pos}")
    print(f"Status 3: {status3}, Pos: {ant3.pos}")

    fig, ax = plt.subplots(3, 1) 
    ax[0].imshow(diff1)
    ax[1].imshow(diff2)
    ax[2].imshow(diff3)
    plt.show()