import os
import cohere

import numpy as np

from typing import List, Dict, Union
from dotenv import load_dotenv

load_dotenv('var.env')
co = cohere.Client(os.environ["COHERE_API_TOKEN"])

def encode_sentences(sentences: Union[str, List[str]]):
    if not isinstance(sentences, List):
        raise ValueError(f"Sentences must be in a List, found type {type(sentences)}")
    results = co.embed(
        model = 'large',
        texts = sentences
    )
    return np.array(results.embeddings)