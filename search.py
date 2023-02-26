import json
import glob
import faiss

import numpy as np

from model import encode_sentences
from typing import List, Tuple, Dict

def load_index(index_path: str):
    return faiss.read_index(index_path)

def load_query_data(data_path: str):
    with open(data_path, "r") as f:
        data = json.load(f)
    data['sentences'] = np.array(data['sentences'])
    data['categories'] = np.array(data['categories'])
    data['urls'] = np.array(data['urls'])
    data['misinf_ratio'] = np.array(data['misinf_ratio'])
    data['n_queries'] = np.array(data['n_queries'])
    return data

def save_query_data(data_path: str, query_data: Dict):
    data['sentences'] = data['sentences'].tolist()
    data['categories'] = data['categories'].tolist()
    data['urls'] = data['urls'].tolist()
    data['misinf_ratio'] = data['misinf_ratio'].tolist()
    data['n_queries'] = data['n_queries'].tolist()
    with open(data_path, "w") as f:
        f.write(json.dumps(query_data))


def query_index(index, query: str, query_data: Dict, k: int = 5) -> Tuple:
    embedding = encode_sentences([query])
    scores, idx = index.search(embedding, k=k)
    misinf_ratios = query_data['misinf_ratio'][idx]
    return query_data['sentences'][idx], query_data['categories'][idx], query_data['urls'][idx], scores, misinf_ratios > 0.8, idx

def get_index_query_files(pattern: str = "search_*"):
    index_pattern, data_pattern = pattern + ".index", pattern + ".json"
    index_files, data_files = glob.glob(index_pattern), glob.glob(data_pattern)

    get_version = lambda s: int(s.split('_')[-1].split('.')[0])
    index_versions, data_versions = [get_version(s) for s in index_files], [get_version(s) for s in data_files]

    index_versions = [i for i in index_versions if i in data_versions]
    data_versions = [d for d in data_versions if d in index_versions]

    max_version = max(max(index_versions), max(data_versions))
    index_path, data_path = f"{pattern[:-1]}{max_version}.index", f"{pattern[:-1]}{max_version}.json"
    return index_path, data_path


if __name__ == "__main__":
    index_path, data_path = get_index_query_files()
    index, data = load_index(index_path), load_query_data(data_path)

    while (query_str := input("Enter the Query String, or enter QUIT to quit: ")).strip().lower() != "quit":
        sentences, categories, urls, scores, misinf, idx = query_index(index, query_str, data)
        good_sentences = sentences[~misinf]
        for i in range(len(good_sentences)):
            print(f"Response {i+1}: ========================================================================")
            print(f"Response: {good_sentences[i]}")
            print(f"Category: {categories[~misinf][i]}")
            print(f"URL: {urls[~misinf][i]}")
            print("=========================================================================================\n")
        good_idx = idx[~misinf]
        new_bad_string = input("Input the Responses that don't seem to be correct as a Comma Separated List: ")
        if new_bad_string != "":
            new_bads = [int(s) - 1 for s in new_bad_string.split(',') if int(s) > 0 and int(s) <= len(good_sentences)]
            new_bad_idxs = idx[~misinf][new_bads]
            data['misinf_ratio'][new_bad_idxs] = (data['misinf_ratio'][new_bad_idxs] * data['n_queries'][new_bad_idxs] + 1) / (data['n_queries'][new_bad_idxs] + 1)
            data['n_queries'][new_bad_idxs] += 1
            print(data['misinf_ratio'][new_bad_idxs])
        
        misinf_sentences = sentences[misinf]
        if len(misinf_sentences) != 0:
            support_samples = []
            for sent in misinf_sentences:
                _, _, _, _, misinf, idx = query_index(index, sent, data)
                support_samples += [idx[~misinf]]
            support_samples = np.concatenate(support_samples)
            print("\n\nThe following queries returned potentially suspicious results:")
            for i in range(len(misinf_sentences)):
                print("=========================================================================================")
                print(f"Response: {misinf_sentences[i]}")
                print(f"Category: {categories[misinf][i]}")
                print(f"URL: {urls[misinf][i]}")
                print("=========================================================================================")
            print("\n\nHere are some sources regarding these bad resources: ")
            for i in support_samples:
                sentence = data['sentences'][i]
                category = data['categories'][i]
                url = data['urls'][i]
                print("=========================================================================================")
                print(f"Response: {sentence}")
                print(f"Category: {category}")
                print(f"URL: {url}")
                print("=========================================================================================\n")
    save_query_data(data_path, data)
    print("Saved Data Sucessfully, Thank You!")




     

