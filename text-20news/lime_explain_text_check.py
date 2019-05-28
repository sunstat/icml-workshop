import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics

from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB

#categories = ['alt.atheism', 'soc.religion.christian']
categories = ['sci.electronics', 'sci.crypt']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
#class_names = ['atheism', 'christian']
class_names = ['eletronics', 'crypt']

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)

pred = rf.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')

pred = rf.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')


from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, rf)

print(c.predict_proba([newsgroups_test.data[2]]))


from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)

def explain(num_iterations):
    #idx = 85
    idx = 103
    features_map = {}
    for i in range(num_iterations):
        exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)
        important_features = exp.as_list()
        for feature, weight in important_features:
            if feature not in features_map:
                features_map[feature] = 0
            features_map[feature] += 1
    return features_map


if __name__ == "__main__":

        features_count = explain(100)
        print(features_count)