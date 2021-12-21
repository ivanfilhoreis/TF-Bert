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

additional 	algorithm 	also 	analysis 	associate 	automatically 	classification 	computer 	create 	datum 	decision 	encompass 	estimate 	experience 	feature 	go 	improve 	input 	involve 	item 	know 	large 	learn 	learning 	machine 	make 	method 	model 	network 	observation 	perform 	prediction 	predictive 	process 	regression 	relate 	relationship 	set 	statistical 	study 	supervise 	support 	train 	training 	tree 	use 	variable 	variety 	vector
0 	0.0741 	0.5081 	0.0107 	0.3017 	0.0535 	0.1425 	0.1458 	0.3725 	0.1057 	0.1657 	0.1077 	0.2035 	0.1675 	0.1827 	0.1222 	-0.0536 	0.2904 	0.1486 	0.0720 	0.0174 	0.0589 	-0.0163 	0.3134 	0.3641 	0.3378 	0.1174 	0.1758 	0.0998 	0.1702 	0.1312 	0.0274 	0.1610 	0.2332 	0.2594 	0.1301 	0.0646 	0.0645 	-0.0032 	0.2902 	0.2972 	0.1820 	-0.0005 	0.1230 	0.2156 	-0.1320 	0.1121 	0.1519 	0.0171 	0.1547
1 	0.2506 	0.4057 	0.1046 	0.5246 	0.1317 	0.0660 	0.2875 	0.1249 	0.0966 	0.3437 	0.2286 	0.3227 	0.3597 	0.1534 	0.1873 	-0.0267 	0.1617 	0.3101 	0.1653 	0.1305 	-0.0114 	0.1262 	0.1260 	0.1705 	0.1115 	0.0988 	0.3434 	0.1349 	0.1587 	0.1298 	0.1227 	0.3459 	0.3845 	0.2668 	0.3516 	0.2342 	0.1397 	0.0775 	0.5280 	0.4095 	0.1288 	0.0461 	0.1317 	0.1499 	0.0599 	0.1547 	0.4039 	0.2788 	0.2597
2 	0.2380 	0.5052 	0.1596 	0.2817 	0.1226 	0.1282 	0.2705 	0.3069 	0.1611 	0.2752 	0.0916 	0.2951 	0.1810 	0.1587 	0.2569 	0.0290 	0.2207 	0.2148 	0.1691 	0.1200 	0.1060 	0.1492 	0.2984 	0.3310 	0.3276 	0.1177 	0.3420 	0.1535 	0.4384 	0.1505 	0.1238 	0.2268 	0.2384 	0.2586 	0.1941 	0.1543 	0.1475 	0.1330 	0.3656 	0.2995 	0.2066 	0.1668 	0.2595 	0.2608 	0.0047 	0.1998 	0.3268 	0.1658 	0.4311
3 	0.1192 	0.4049 	0.0041 	0.3208 	0.1763 	0.1256 	0.2286 	0.1265 	0.1305 	0.2829 	0.4078 	0.1868 	0.3297 	0.1763 	0.1445 	0.0584 	0.1551 	0.1740 	0.1711 	0.1414 	0.1711 	-0.0130 	0.2735 	0.2960 	0.1418 	0.1375 	0.1452 	0.1772 	0.1577 	0.2122 	0.0014 	0.4270 	0.4888 	0.2140 	0.1558 	0.1920 	0.1565 	0.0978 	0.3023 	0.3222 	0.1195 	0.0383 	0.1312 	0.1177 	0.2511 	0.1356 	0.2601 	0.2315 	0.1824
4 	0.1428 	0.4573 	0.0404 	0.3019 	0.0918 	0.1633 	0.2108 	0.2585 	0.2100 	0.2770 	0.1314 	0.2244 	0.3032 	0.2218 	0.1391 	-0.0800 	0.1696 	0.1938 	0.1077 	0.0778 	0.0735 	0.0374 	0.3193 	0.3496 	0.3642 	0.1950 	0.2553 	0.2861 	0.1744 	0.1835 	0.1432 	0.3139 	0.3606 	0.3068 	0.1376 	0.1095 	0.0631 	0.0857 	0.3529 	0.3283 	0.2036 	-0.0128 	0.2598 	0.3277 	-0.0132 	0.1264 	0.2674 	0.0979 	0.1892

```


