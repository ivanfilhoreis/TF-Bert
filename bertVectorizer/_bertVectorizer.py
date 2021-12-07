import pandas as pd
import numpy as np
import re
#from PreProcessing import Text_PreProcessing, Stemming
from sklearn.feature_extraction.text import CountVectorizer
from keybert import KeyBERT
from typing import List, Union, Tuple
from tqdm import tqdm


class bertVectorizer():
    def __init__(self,
                 stemmer: bool = False,
                 lowercase: bool = True,
                 ngram_range: Tuple = (1, 1),
                 stop_words=None,
                 max_df: float = 1.0,
                 min_df: float = 1,
                 max_features:  int = None,
                 vocabulary: Union[str, List[str]] = None,
                 kbert_candidates: List[str] = None,
                 kbert_model: str = 'distilbert-base-nli-mean-tokens',
                 kbert_top_n: int = 5,
                 kbert_min_df: int = 1,
                 kbert_use_maxsum: bool = False,
                 kbert_use_mmr: bool = False,
                 kbert_diversity: float = 0.5,
                 kbert_nr_candidates: int = 20,
                 kbert_vectorizer: CountVectorizer = None,
                 kbert_highlight: bool = False,
                 kbert_seed_keywords: List[str] = None,
                 normalize: bool = True) -> None:

        self.stemmer = stemmer
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary = vocabulary
        self.kbert_candidates = kbert_candidates
        self.kbert_model = kbert_model
        self.kbert_top_n = kbert_top_n
        self.kbert_min_df = kbert_min_df
        self.kbert_use_maxsum = kbert_use_maxsum
        self.kbert_use_mmr = kbert_use_mmr
        self.kbert_diversity = kbert_diversity
        self.kbert_nr_candidates = kbert_nr_candidates
        self.kbert_vectorizer = kbert_vectorizer
        self.kbert_highlight = kbert_highlight
        self.kbert_seed_keywords = kbert_seed_keywords
        self.normalize = normalize

    def fit_transform(self, df):
        if self.vocabulary == 'keybert':
            self.vocabulary = self.features_keybert(df)
        else:
            cv_vectorizer = CountVectorizer(lowercase=self.lowercase,
                                            stop_words=self.stop_words,
                                            ngram_range=self.ngram_range,
                                            max_df=self.max_df,
                                            min_df=self.min_df,
                                            max_features=self.max_features,
                                            vocabulary=self.vocabulary)

            cv_trained = cv_vectorizer.fit_transform(df)
            self.vocabulary = cv_trained.get_feature_names()

        keybert_model = KeyBERT(model=self.kbert_model)
        weights_bert = []

        for item in df:
            items = []
            keybert_result = sorted(keybert_model(item,
                                                  candidates=self.vocabulary,
                                                  top_n=len(df)))  # extract similarity

            for item in keybert_result:
                items.append(item[1])

            weights_bert.append(items)
        weights_bert = np.array(weights_bert)

        df_bert = pd.DataFrame(columns=self.vocabulary, data=weights_bert)

        return df_bert

    def features_keybert(self, text):

        kw_model = KeyBERT(self.kbert_model)
        keys = list()
        result = 0

        for txt, prog in zip(text, tqdm(range(0, len(text)))):
            result += prog
            if txt == 'no texts':
                continue

            keywords = kw_model.extract_keywords(txt,
                                                 candidates=self.kbert_candidates,
                                                 keyphrase_ngram_range=self.ngram_range,
                                                 stop_words=self.stop_words,
                                                 top_n=self.kbert_top_n,
                                                 min_df=self.kbert_min_df,
                                                 use_maxsum=self.kbert_use_maxsum,
                                                 use_mmr=self.kbert_use_mmr,
                                                 diversity=self.kbert_diversity,
                                                 nr_candidates=self.kbert_nr_candidates,
                                                 vectorizer=self.kbert_vectorizer,
                                                 highlight=self.kbert_highlight,
                                                 seed_keywords=self.kbert_seed_keywords)

            keys.append(keywords)
            # print(keywords)

        vocab = []
        for k in keys:
            for (kw, vl) in k:
                vocab.append(kw)

        vocabulary = set(d for d in vocab)

        print(vocabulary)

        return list(vocabulary)
