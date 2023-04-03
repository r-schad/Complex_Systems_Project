import math
import argparse

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from network import LatticeNetwork
from tqdm import tqdm
from typing import Tuple

rng = np.random.default_rng(seed=0)

class Ant:
    def __init__(self, vec: np.ndarray, pos: tuple, alpha: float, beta: float, delta: float, 
                 eps: float = 0.01, move_base: float = 2.0):
        # initialize ant with the document vector
        self.vec = vec
        self.pos = pos
        self.prev_pos = None
        self.eps = eps

        # initialize hyperparams
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.move_base = move_base

    def get_move_diff(self, candidate: Tuple):
        if self.prev_pos is None:
            return 1.0
        curr_move = tuple(candidate[i] - self.pos[i] for i in range(len(self.pos)))
        prev_move = tuple(self.pos[i] - self.prev_pos[i] for i in range(len(self.pos)))
        angle_diff = math.atan2(*curr_move) - math.atan2(*prev_move)
        return self.move_base ** (-abs(angle_diff))

    def decide_next_position(self, network: LatticeNetwork, q: float = 0.2) -> bool:
        # compute neighbors and corresponding pheromone levels
        neighbors = network.get_neighbors(*self.pos)
        centroid_vecs = [network.get_centroid_pheromone_vec(r, c) for r, c in neighbors]
        pheromone_vecs = [network.get_pheromone_vec(r, c) for r, c in neighbors]
        edge_pheromones = [self.find_edge_pheromone(centroid_vecs[i], pheromone_vecs[i]) for i in range(len(neighbors))]
        pheromones = [self.pheromone_weighting(sigma) for sigma in edge_pheromones]
        # compute the pheremone scalars for the change in directions
        move_diffs = [self.get_move_diff(n) for n in neighbors]
        pheromones = [move_diffs[i] * p for i, p in enumerate(pheromones)]

        # compute current node pheromones
        current_pheromone = self.find_edge_pheromone(network.get_centroid_pheromone_vec(*self.pos), 
                                                     ant.vec)
        current_pheromone = self.pheromone_weighting(current_pheromone)

        stopped = False
        if rng.uniform() < q:
            # take the greedy option with probability q
            i = np.argmax(pheromones)
            if pheromones[i] < current_pheromone:
                stopped = True
        else:
            # TODO: Figure out if stochastic stop is necessary
            probs = np.array(pheromones) / np.sum(pheromones)
            i = int(self.roulette_wheel(probs))
        if not stopped:
            new_pos = neighbors[i]
            # TODO: Implement previous move tracking
            self.prev_pos = self.pos
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
    args.add_argument("-r", "--centroid-radius", type=int, default=1)
    args.add_argument("-b", "--beta", type=float, default=32)
    args.add_argument("-d", "--delta", type=float, default=0.2)
    args.add_argument("-e", "--embedding-dim", type=int, default=5)
    args.add_argument("-c", "--num-classes", type=int, default=5)
    args.add_argument("-v", "--evaporation-factor", type=float, default=0.99)
    args.add_argument("-s", "--num-steps", type=int, default=600)
    args.add_argument("-m", "--warmup-steps", type=int, default=200)
    args.add_argument("-q", "--greedy-prob", type=float, default=0.2)
    args.add_argument("-z", "--zeros", action='store_true')
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()
    network = LatticeNetwork((args.width, args.width), args.embedding_dim, args.evaporation_factor, 
                             rng=rng, centroid_radius=args.centroid_radius, zeros=args.zeros)
    existing_locs = set()
    ants, status = [], []
    vec_set = np.eye(args.num_classes, args.embedding_dim)
    ant_set = [[] for _ in range(args.num_classes)]
    for i in range(args.num_ants):
        j = rng.choice(args.num_classes)
        ant_vec = vec_set[j] + rng.normal(scale=0.2, size=args.embedding_dim)
        while True:
            loc = tuple(rng.choice(np.arange(args.width), 2))
            if loc not in existing_locs:
                existing_locs.add(loc)
                break
        ants += [(i, Ant(ant_vec, loc, 1, args.beta, args.delta))]
        ant_set[j] += [i]
        status += [False]

    for i, s in enumerate(status):
        print(f"Starting Status {i}: {status[i]}, Pos: {ants[i][1].pos}")

    # run ACO self organization
    for i in tqdm(range(args.num_steps)):
        rng.shuffle(ants)
        for j, ant in ants:
            network.deposit_pheromone(ant.vec, *ant.pos)
            if not status[j] and i > args.warmup_steps:
                status[j] = ant.decide_next_position(network, args.greedy_prob)
            elif i <= args.warmup_steps:
                ant.decide_next_position(network, args.greedy_prob)
        network.evaporate_pheromones()
        if all(status):
            break

    for i, ant in ants:
        final_pheromone_vec = network.get_pheromone_vec(*ant.pos)
        pheromone = ant.find_edge_pheromone(final_pheromone_vec, ant.vec)
        print(f"Status {i}: {status[i]}, Pos: {ant.pos}, Final Pheromone: {pheromone}")

    diffs = [ants[0][1].pheromone_weighting(np.linalg.norm(network.pheromones - vec, axis=-1)) for vec in vec_set]
    min_diffs = [np.min(diff) for diff in diffs]
    for j, ant_idxs in enumerate(ant_set):
        class_ants = [ant for idx, ant in ants if idx in ant_idxs]
        for ant in class_ants:
            r, c = ant.pos
            diffs[j][r, c] = min_diffs[j] / 2

    fig, ax = plt.subplots(1, args.num_classes) 
    for i, diff in enumerate(diffs):
        ax[i].imshow(diff)
    plt.show()

    new_ant_class = int(input("Provide New Ant Class: "))
    new_pos = tuple(rng.choice(np.arange(args.width), 2))
    new_ant = Ant(vec_set[new_ant_class], new_pos, 1, args.beta, args.delta)

    pos_seq = []
    while True:
        pos_seq += [new_ant.pos]
        status = new_ant.decide_next_position(network, q=args.greedy_prob)
        pheromone = new_ant.find_edge_pheromone(network.get_pheromone_vec(*new_ant.pos), new_ant.vec)
        if pheromone > (0.8 * np.max(diffs[new_ant_class])):
            break

    final_match = new_ant.find_edge_pheromone(network.get_pheromone_vec(*new_ant.pos), new_ant.vec)
    print(f"Path Length: {len(pos_seq)}")
    print(f"Final Match: {final_match}")
    path_diff = ants[0][1].pheromone_weighting(np.linalg.norm(network.pheromones - vec_set[new_ant_class], axis=-1))
    diff_min = np.min(path_diff)
    base = diff_min / 2
    for i, pos in enumerate(pos_seq):
        r, c = pos
        path_diff[r, c] = base + ((diff_min - base) * (i / len(pos_seq)))
    plt.imshow(path_diff)
    plt.show()