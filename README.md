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

The most minimal example can be seen below:

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
>>> matrix.iloc[:, 0:15] # It presents columns from 0 to 15. 

index	additional	algorithm	also	analysis	associate	automatically	classification	computer	create	datum	decision	encompass	estimate	experience	feature
0	0.0741		0.5081		0.0107	0.3017		0.0535		0.1425		0.1458		0.3725		0.1057	0.1657	0.1077		0.2035		0.1675		0.1827		0.1222
1	0.2506		0.4057		0.1046	0.5246		0.1317		0.0660		0.2875		0.1249		0.0966	0.3437	0.2286		0.3227		0.3597		0.1534		0.1873
2	0.2380		0.5052		0.1596	0.2817		0.1226		0.1282		0.2705		0.3069		0.1611	0.2752	0.0916		0.2951		0.1810		0.1587		0.2569
3	0.1192		0.4049		0.0041	0.3208		0.1763		0.1256		0.2286		0.1265		0.1305	0.2829	0.4078		0.1868		0.3297		0.1763		0.1445
4	0.1428		0.4573		0.0404	0.3019		0.0918		0.1633		0.2108		0.2585		0.2100	0.277	0.1314		0.2244		0.3032		0.2218		0.1391
```

You can also use the `n_grams` parameter to choose the size of features.

```python
>>> vectorizer = bertVectorizer(n_grams=3)
>>> matrix = vectorizer.fit_transform(df)
>>> matrix.iloc[:, 0:5] # It presents columns from 0 to 5. 

index	additional datum make	algorithm improve automatically	also know support	analysis encompass large	automatically experience use
0	0.1826			0.6178					0.0532			0.2133				0.3264
1	0.3270			0.2543					0.0613			0.5349				0.1611
2	0.2877			0.4454					0.2086			0.2969				0.2459
3	0.2383			0.3078					0.0775			0.1996				0.2412
4	0.2899			0.4197					0.0285			0.2368				0.2773

```

