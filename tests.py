import numpy as np
import matplotlib.pyplot as plt
import argparse
from network import LatticeNetwork, inv_cos_dist
from ant import Ant
from model import load_model
from functools import reduce
from main import ant_search

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("-w", "--width", type=int, default=100)
    args.add_argument("-n", "--num-ants", type=int, default=1000)
    args.add_argument("-b", "--beta", type=float, default=32)
    args.add_argument("-d", "--delta", type=float, default=0.2)
    args.add_argument("-k", "--reinforce-exp", type=float, default=3)
    args.add_argument("-i", "--input-data", type=str, default='test_embed.pkl')
    args.add_argument("-s", "--num-steps", type=int, default=200)
    args.add_argument("-v", "--evaporation-factor", type=float, default=0.995)
    args.add_argument("-r", "--centroid-radius", type=int, default=1)
    args.add_argument("-z", "--zeros", action='store_true')
    args.add_argument("-q", "--greedy-prob", type=float, default=0.2)
    args.add_argument("-m", "--warmup-steps", type=int, default=0)
    args.add_argument("-e", "--export-video", action='store_true')
    return args.parse_args()

if __name__ == '__main__':
    rng = np.random.default_rng(seed=0)
    args = parse_args()

    # import network and its documents from pkl
    acses_network = LatticeNetwork.from_pickle('final_network.pkl')
    all_docs = np.vstack(reduce(lambda a,b: a+b, acses_network.documents.flatten())).flatten()
    all_doc_vecs = reduce(lambda a,b: a+b, acses_network.doc_vecs.flatten())

    model = load_model()

    random_placements = np.random.randint(low=0, high=args.width, size=(all_docs.shape[0], 2))

    # taken from https://stackoverflow.com/questions/43483663/how-do-i-make-a-grid-of-empty-lists-in-numpy-that-accepts-appending#:~:text=m%20%3D%20np.empty((12%2C%2012)%2C%20dtype%3Dobject)%0Afor%20i%20in%20np.ndindex(m.shape)%3A%20m%5Bi%5D%20%3D%20%5B%5D
    basic_network_docs = np.empty((args.width, args.width), dtype=object)
    for i in np.ndindex(basic_network_docs.shape): basic_network_docs[i] = []

    basic_network_vecs = np.ndarray((args.width,args.width), dtype=list)
    for i in np.ndindex(basic_network_vecs.shape): basic_network_vecs[i] = []

    for i in range(all_docs.shape[0]):
        basic_network_docs[random_placements[i,0]][random_placements[i,1]].append(all_docs[i])
        basic_network_vecs[random_placements[i,0]][random_placements[i,1]].append(all_doc_vecs[i])

    doc_idxs = []

    # iterate through N sample documents,
    # calculate the inv_cos_dist from all docs at a randomly chosen node for the doc
    # calculate the inv_cod_dist from all docs at a ACSeS chosen node for the doc
    baseline_means = []
    baseline_mins = []
    baseline_maxs = []

    acses_means = []
    acses_mins = []
    acses_maxs = []
    # sample N documents and compare ACSeS vs random allocation 
    while len(doc_idxs) < args.num_ants:
        # get a random document index
        doc_idx = np.random.randint(low=0, high=all_docs.shape[0])

        # using baseline (random allocation)

        # get a random node from the 100x100 grid
        baseline_node_idx = np.random.randint(low=0, high=args.width, size=2)
        # if that node doesn't have any documents, skip this document
        if len(basic_network_docs[baseline_node_idx[0], baseline_node_idx[1]]) == 0: continue

        doc_idxs += [doc_idx]

        # get the document and vectors stored in the randomly selected node
        baseline_docs = basic_network_docs[baseline_node_idx[0], baseline_node_idx[1]]
        baseline_vecs = basic_network_vecs[baseline_node_idx[0], baseline_node_idx[1]]

        # calculate the inverse cosine distance of the current doc with all docs in the randomly selected node
        baseline_dists = [inv_cos_dist(all_doc_vecs[doc_idx], vec) for vec in baseline_vecs]
        if baseline_dists == []: baseline_dists = [2.0]

        # store baseline stats
        baseline_means += [np.mean(baseline_dists)]
        baseline_mins += [np.min(baseline_dists)]
        baseline_maxs += [np.max(baseline_dists)]

        # using ACSeS

        # initialize ant
        ant_vec = all_doc_vecs[doc_idx]
        loc = tuple(rng.choice(np.arange(args.width), 2))
        ant = Ant(ant_vec, loc, 1, args.beta, args.delta, reinforce_exp=args.reinforce_exp, ant_id=i, document=all_docs[doc_idx])

        ant, acses_docs, pos_seq, pher_seq = ant_search(acses_network, ant, q=args.greedy_prob)

        # get the vectors stored in the ACSeS-selected node
        acses_node_idx = pos_seq[-1]
        acses_vecs = acses_network.doc_vecs[acses_node_idx[0]][acses_node_idx[1]]

        # calculate the inverse cosine distance of the current doc with all docs in the randomly selected node
        acses_dists = [inv_cos_dist(all_doc_vecs[doc_idx], vec) for vec in acses_vecs]

        # store acses stats
        acses_means += [np.mean(acses_dists)]
        acses_mins += [np.min(acses_dists)]
        acses_maxs += [np.max(acses_dists)]
   
    f,axs = plt.subplots(2)

    axs[0].scatter(range(len(doc_idxs)), baseline_means, c='r', label='baseline_means')
    axs[0].scatter(range(len(doc_idxs)), baseline_mins, c='m', label='baseline_mins')
    axs[0].scatter(range(len(doc_idxs)), baseline_maxs, c='y', label='baseline_maxs')
    axs[0].legend()

    axs[1].scatter(range(len(doc_idxs)), acses_means, c='b', label='acses_means')
    axs[1].scatter(range(len(doc_idxs)), acses_mins, c='c', label='acses_mins')
    axs[1].scatter(range(len(doc_idxs)), acses_maxs, c='g', label='acses_maxs')

    axs[1].legend()

    plt.show()
    pass


    
    
