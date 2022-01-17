# -*- coding: utf-8 -*-
# Authors: Ivan José dos Reis Filho <ivanfilhoreis@gmail.com>
#          Luiz Henrique Dutra Martins <luizmartins.uemg@gmail.com>

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
    r"""

    """

    def __init__(self,
                bert_model='nli-distilroberta-base-v2',
                n_grams=1, 
                clear_texts=True,) -> None:
        
        self.bert_model = bert_model
        self.n_grams = n_grams
        self.clear_texts = clear_texts
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.model = SentenceTransformer(self.bert_model)
       
    def preprocess_candidates(self, text):
        """[summary]

        Args:
            text([type]): [description]

        Returns:
            [type]: [description]
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
        ", ".join(stopwords.words('english'))
        STOPWORDS = set(stopwords.words('english'))

        text = " ".join([word for word in str(
            text).split() if word not in STOPWORDS])

        # remove mentions and hashtags
        text = re.sub(r'@\w+', '', text)

        # remove hash
        text = re.sub(r'#\w+', '',text)

        # Remove extra white space left while removing stuff
        text = re.sub(r"\s+", " ", text).strip()
        
        #remove punctuation 
        text = " ".join([word for word in str(text).split() if word not in string.punctuation])

        return text
    
    def generate_ngrams(self, text):
        """
        Parameters
        ----------
        text : TYPE
            DESCRIPTION.
        n_gram : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        list
            DESCRIPTION.

        """
        token = [token for token in text.split(' ') if token != '']
        
        ngrams = zip(*[token[i:] for i in range(self.n_grams)])
        
        return [' '.join(ngram) for ngram in ngrams]


    def get_features(self, data):
        """[summary]

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        data['clean_text'] = data.text.apply(lambda text: self.preprocess_candidates(text))
        
        candidates = set()
        
        if self.clear_texts is False:
            data.clean_text = data.text

        for item in data.clean_text:
            doc = self.nlp(item)
            new_sentence = [token.lemma_ for token in doc if token.is_alpha]
            new_sentence = ' '.join(new_sentence)
            
            for words in self.generate_ngrams(new_sentence):
                candidates.add(words)
        
            
        return sorted(candidates)
    
    
    
    def encode_data(self, data, candidates):
        """[summary]

        Args:
            data ([type]): [description]
            candidates ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        model = self.model

        emb_data = model.encode(data)
        emb_candidates = model.encode(candidates)

        return emb_data, emb_candidates

    def fit_transform(self, data):
        """[summary]

        Args:
            data ([type]): [description]

        Returns:
            [Pandas DataFrame]: [description]
        """
        candidates = self.get_features(data)
        emb_data, emb_candidates = self.encode_data(data.text, candidates)
        
        
        matrix = []
        
        for index in range(len(emb_data)):
            text_similarity = cosine_similarity([emb_data[index]], emb_candidates[0:])
            
            matrix.append(text_similarity[0])

        dataframe = pd.DataFrame(columns=candidates, data=matrix)
        
        return dataframe
