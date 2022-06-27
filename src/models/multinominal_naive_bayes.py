from sklearn.naive_bayes import (
    BernoulliNB,  # For multivariate models and suiable for discrete data
    ComplementNB,  # Used to correct 'severe assumptions made by Multinominal classifier
    MultinomialNB,  # Suitable for classification with discrete features, can work with td-idf but made for integer counts ideally
    CategoricalNB,  # For categorical geatures with discrete features
)

classifiers = {
    "MultinominmalNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB(),
    "CategoricalNB": CategoricalNB(),
}