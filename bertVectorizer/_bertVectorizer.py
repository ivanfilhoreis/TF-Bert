# -*- coding: utf-8 -*-
# Authors: Ivan José dos Reis Filho <ivanfilhoreis@gmail.com>
#          Luiz Henrique Dutra Martins <luizmartins.uemg@gmail.com>

import pandas as pd
import nltk 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

__all__ = ['bertVectorizer']


class bertVectorizer():
    r"""

    """
    def __init__(self,
                bert_model='nli-distilroberta-base-v2') -> None:
        self.bert_model = bert_model

    def encode_data(self, data, candidates):
        """[summary]

        Args:
            data ([type]): [description]
            candidates ([type]): [description]

        Returns:
            [type]: [description]
        """
        model = SentenceTransformer(self.bert_model)

        emb_data = model.encode(data)
        emb_candidates = model.encode(candidates)

        return emb_data, emb_candidates

    def get_features(self, data):
        """[summary]

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """
        candidates = []

        return candidates

    def fit_transform(self, data):
        """[summary]

        Args:
            data ([type]): [description]

        Returns:
            [Pandas DataFrame]: [description]
        """
        matrix = []

        candidates = self.get_features(data)

        return pd.DataFrame(columns=candidates, data=matrix)
