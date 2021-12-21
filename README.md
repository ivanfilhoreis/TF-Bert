# bertVectorizer

Convert a collection of raw documents to a matrix extracted from BERT resources. 

This implementation produces a non-sparse representation of the text's resource count.

# 1. Installation

Installation and use can be done as:

```python
!pip install git+https://github.com/ivanfilhoreis/bertVectorizer.git -q
!pip install -U sentence-transformers -q
```

# 2. Usage

```python
import pandas as pd

dic = {
       'text': ['Machine learning is the study of computer algorithms that can improve automatically through experience and by the use of data',
                'Regression analysis encompasses a large variety of statistical methods to estimate the relationship between input variables and their associated features',
                'Support-vector machines also known as support-vector networks are a set of related supervised learning methods used for classification and regression',
                'Decision tree learning uses a decision tree as a predictive model to go from observations about an item',
                'Performing machine learning involves creating a model which is trained on some training data and then can process additional data to make predictions']}
            
df = pd.DataFrame(dic)

```

You can use `fit_transform` from `bertVectorizer` to convert a collection of raw documents to a matrix extracted from BERT resources:

```python
>>> from bertVectorizer import bertVectorizer
>>> vectorizer = bertVectorizer()
>>> matrix = vectorizer.fit_transform(df)

index	additional	algorithm	also	analysis	associate	automatically	classification	computer	create	datum	decision	encompass	estimate	experience	feature	go	improve	input	involve	item
0	0.074	0.508	0.011	0.302	0.053	0.143	0.146	0.372	0.106	0.166	0.108	0.203	0.168	0.183	0.122	-0.054	0.29	0.149	0.072	0.017
1	0.251	0.406	0.105	0.525	0.132	0.066	0.287	0.125	0.097	0.344	0.229	0.323	0.36	0.153	0.187	-0.027	0.162	0.31	0.165	0.13
2	0.238	0.505	0.16	0.282	0.123	0.128	0.27	0.307	0.161	0.275	0.092	0.295	0.181	0.159	0.257	0.029	0.221	0.215	0.169	0.12
3	0.119	0.405	0.004	0.321	0.176	0.126	0.229	0.127	0.131	0.283	0.408	0.187	0.33	0.176	0.144	0.058	0.155	0.174	0.171	0.141
4	0.143	0.457	0.04	0.302	0.092	0.163	0.211	0.258	0.21	0.277	0.131	0.224	0.303	0.222	0.139	-0.08	0.17	0.194	0.108	0.078

```


