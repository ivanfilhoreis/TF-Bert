# -*- coding: utf-8 -*-
# Authors: Ivan José dos Reis Filho <ivanfilhoreis@gmail.com>
#          Luiz Henrique Dutra Martins <luizmartins.uemg@gmail.com>

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keybert import KeyBERT
from typing import List, Union, Tuple
from tqdm import tqdm

__all__ = ['bertVectorizer']


class bertVectorizer():
    r"""Convert a collection of raw documents to a matrix extracted from BERT resources. 

    This library is based on the KeyBERT and CountVectorizer libraries. We recommend that you read the documentation for both. 

    Parameters
    ----------

    stemmer : bool, default=False
        Performs stemming in the text 

    lowercase : bool, default=True
        Convert all characters to lowercase before tokenizing

    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        word n-grams or char n-grams to be extracted. All values of n such
        such that min_n <= n <= max_n will be used. For example an
        ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
        unigrams and bigrams, and ``(2, 2)`` means only bigrams.
        Only applies if ``analyzer is not callable``.

    stop_words : {'english'}, list, default=None
        If 'english', a built-in stop word list for English is used.
        There are several known issues with 'english' and you should
        consider an alternative (see :ref:`stop_words`).

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : int, default=None
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable, default=None
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.

    kbert_candidates : List, default=None 
        Candidate keywords/keyphrases to use instead of extracting them from the document(s)

    kbert_model : String, default='distilbert-base-nli-mean-tokens.
        Select a BERT model

    kbert_top_n : int, default=5
        Return the top n keywords/keyphrases

    kbert_min_df : int, default=1
        Minimum document frequency of a word across all documents if keywords for multiple documents need to be extracted

    kbert_use_maxsum : bool, default=False
        Whether to use Max Sum Similarity for the selection of keywords/keyphrases

    kbert_use_mmr : bool, default=False
        Whether to use Maximal Marginal Relevance (MMR) for the selection of keywords/keyphrases

    kbert_diversity : float, default=0.5
        The diversity of the results between 0 and 1 if use_mmr is set to True

    kbert_nr_candidates : int, default=20
        The number of candidates to consider if use_maxsum is set to True

    kbert_vectorizer : CountVectorizer, default=None
        Pass in your own CountVectorizer from scikit-learn

    kbert_highlight : bool, default=False
        Whether to print the document and highlight its keywords/keyphrases. NOTE: This does not work if multiple documents are passed. 

    kbert_seed_keywords : list, default=None
        Seed keywords that may guide the extraction of keywords by steering the similarities towards the seeded keywords

    """

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
            self.vocabulary = cv_vectorizer.get_feature_names()

        keybert_model = KeyBERT(model=self.kbert_model)
        weights_bert = []

        for item in df:
            items = []
            keybert_result = sorted(keybert_model.extract_keywords(
                item, candidates=self.vocabulary, top_n=len(self.vocabulary)))  # extract similarity

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
