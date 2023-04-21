import numpy as np
from network import LatticeNetwork
from ant import Ant
from model import load_model
from functools import reduce

if __name__ == '__main__':

    # import network and its documents from pkl
    acses_network = LatticeNetwork.from_pickle('final_network.pkl')
    all_docs = np.vstack(reduce(lambda a,b: a+b, acses_network.documents.flatten())).flatten()
    all_doc_vecs = reduce(lambda a,b: a+b, acses_network.doc_vecs.flatten())

    model = load_model()

    # dear god why
    closenesses = [np.sum(np.isclose(model.encode(all_docs[i]), all_doc_vecs[i])) for i in range(all_docs.shape[0])] 

    # store all documents randomly a normal 100x100 lattice
    # random_placements = np.random.randint(0, 100, size=(all_docs.shape))




