from _bertVectorizer import bertVectorizer


text = ['Machine learning is the study of computer algorithms that can improve automatically through experience and by the use of data',
        'Regression analysis encompasses a large variety of statistical methods to estimate the relationship between input variables and their associated features',
        'Support-vector machines also known as support-vector networks are a set of related supervised learning methods used for classification and regression',
        'Decision tree learning uses a decision tree as a predictive model to go from observations about an item',
        'Performing machine learning involves creating a model which is trained on some training data and then can process additional data to make predictions']

words = ['study', 'computer', 'algorithms', 'machines', 'networks', 'predictive', 'learning', 'tree', 'additional', 'data', 'regression', 'classification', 'supervised']

vectorizer = bertVectorizer(stp_wrds=True, candidates=words)

matrix = vectorizer.fit_transform(text)

print(matrix)

