import argparse

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

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("-n", "--num-ants", type=int, default=5)
    args.add_argument("-w", "--width", type=int, default=5)
    args.add_argument("-b", "--beta", type=float, default=32)
    args.add_argument("-d", "--delta", type=float, default=0.2)
    args.add_argument("-e", "--embedding-dim", type=int, default=5)
    args.add_argument("-v", "--evaporation-factor", type=float, default=0.99)
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()
    network = LatticeNetwork((args.width, args.width), args.embedding_dim, 0.99, rng=rng)
    existing_locs = set()
    ants, status = [], []
    for i in range(args.num_ants):
        ant_vec = np.zeros(args.embedding_dim)
        ant_vec[i] = 1
        while True:
            loc = tuple(rng.choice(np.arange(args.width), 2))
            if loc not in existing_locs:
                break
        ants += [Ant(ant_vec, loc, 1, args.beta, args.delta)]
        status += [False]

    for i, s in enumerate(status):
        print(f"Starting Status {i}: {status[i]}, Pos: {ants[i].pos}")

    i = 0
    while not any(status) and i < 10000:
        for ant in ants:
            network.deposit_pheromone(ant.vec, *ant.pos)
            ant.decide_next_position(network, 0.5)
        network.evaporate_pheromones()
        i += 1

    diffs = [ant.pheromone_weighting(np.linalg.norm(network.pheromones - ant.vec, axis=-1)) for ant in ants]
    for i, s in enumerate(status):
        print(f"Status {i}: {status[i]}, Pos: {ants[i].pos}")

    fig, ax = plt.subplots(args.num_ants, 1) 
    for i, diff in enumerate(diffs):
        ax[i].imshow(diff)
    plt.show()