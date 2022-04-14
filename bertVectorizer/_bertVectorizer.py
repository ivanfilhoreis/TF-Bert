# -*- coding: utf-8 -*-
# Authors: Ivan José dos Reis Filho <ivanfilhoreis@gmail.com>
#          Luiz Henrique Dutra Martins <luizmartins.uemg@gmail.com>

from multiprocessing.sharedctypes import Value
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import re
import spacy
import string

nltk.download('stopwords')

__all__ = ['bertVectorizer']


class bertVectorizer():
    r"""Convert a collection of text documents to a dataframe of token similarity.

    The algorithm uses a BERT model as a base and to find the similarity between features and text it uses the cosine similarity function. 

    Parameters 
    ----------

    bert_model : str, default='nli-distilroberta-base-v2'
        Select bert model to encode data. You can use a transformer model from the sentence-transformers library

    n_grams : int, default=1
        Inform n_grams to choose the size of features

        You can enter any value greater than zero, the algorithm only generates individual ngrams. 

    clear_texts : bool, default=True
        Performs a pre-processing on the texts to get the features. 
    """

    def __init__(self,
                 bert_model='nli-distilroberta-base-v2',
                 spacy_lang='en_core_web_sm',
                 lang='english',
                 n_grams=1,
                 stp_wrds=True,
                 all_features=True
                 ) -> None:

        self.bert_model = bert_model
        self.n_grams = n_grams
        self.lang = lang
        self.stp_wrds = stp_wrds
        self.stp_wrds = stp_wrds
        self.all_features = all_features
        self.nlp = spacy.load(spacy_lang, disable=['parser', 'ner'])
        self.model = SentenceTransformer(self.bert_model)

    def preprocess_candidates(self, text):
        """[Performs a pre-processing on the text]

        Args:
            text([str]): [receive a text as a string] 

        Returns:
            [str]: [returns pre-processed text]
        """

        # urls
        url_remove = re.compile(r'https?://\S+|www\.\S+')
        text = url_remove.sub(r'', text)

        # html
        html_remove = re.compile(r'<.*?>')
        text = html_remove.sub(r'', text)

        # lowercase
        text = text.lower()

        # number removal
        text = re.sub(r'\d+', '', text)

        # remove stopwords
        ", ".join(stopwords.words(self.lang))
        STOPWORDS = set(stopwords.words(self.lang))

        text = " ".join([word for word in str(
            text).split() if word not in STOPWORDS])

        # remove mentions and hashtags
        text = re.sub(r'@\w+', '', text)

        # remove hash
        text = re.sub(r'#\w+', '', text)

        # Remove extra white space left while removing stuff
        text = re.sub(r"\s+", " ", text).strip()

        # remove punctuation
        text = " ".join([word for word in str(text).split()
                        if word not in string.punctuation])

        return text

    def generate_ngrams(self, text):
        """[generating ngrams from text]

        Args:
            text ([str]): [receive a text as a string]

        Returns:
            [list]: [returns a list of ngrams from the text]
        """
        token = [token for token in text.split(' ') if token != '']

        ngrams = zip(*[token[i:] for i in range(self.n_grams)])

        return [' '.join(ngram) for ngram in ngrams]

    def get_features(self, data):
        """[Get the features from data]

        Args:
            data ([pandas dataframe]): [receive a dataframe from pandas]

        Returns:
            [set]: [returns a set of the features]
        """
        if isinstance(data, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )
        else:
            try:
                data = list(data)
            except ValueError:
                print("Data type is invalid.")

        stp_wrds_clear = []

        if data.__class__.__name__ == 'list':
            for text in data:
                stp_wrds_clear.append(self.preprocess_candidates(text))

        candidates = set()

        if self.stp_wrds is False:
            document = data
        else:
            document = stp_wrds_clear

        for item in document:
            doc = self.nlp(item)
            new_sentence = [token.lemma_ for token in doc if token.is_alpha]
            new_sentence = ' '.join(new_sentence)

            for words in self.generate_ngrams(new_sentence):
                candidates.add(words)

        return sorted(candidates)

    def encode_data(self, data, candidates=None):
        """[Encode data using BERT]

        Args:
            data ([pandas dataframe]): [receive a dataframe from pandas]
            candidates ([set]): [receive a set with the features]

        Returns:
            [array numpy]: [returns two arrays from numpy, the first has the encoding of texts and the second has the encoding of features]
        """
        if isinstance(data, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )

        if candidates == None:
            raise ValueError(
                "A list of candidates was expected, None object received."
            )

        model = self.model

        emb_data = model.encode(data)
        emb_candidates = model.encode(candidates)

        return emb_data, emb_candidates

    def fit_transform(self, data):
        """[Transform a sequence of documents to a similarity document dataframe]

        Args:
            data ([pandas dataframe]): [receive a dataframe from pandas]

        Returns:
            [Pandas DataFrame]: [returns a pandas dataframe containing the similarity of the terms with the texts]
        """
        if isinstance(data, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )
        candidates = self.get_features(data)
        emb_data, emb_candidates = self.encode_data(data, candidates)

        matrix = []

        for index in range(len(emb_data)):
            text_similarity = cosine_similarity(
                [emb_data[index]], emb_candidates[0:])

            matrix.append(text_similarity[0])

        dataframe = pd.DataFrame(columns=candidates, data=matrix)

        return dataframe
