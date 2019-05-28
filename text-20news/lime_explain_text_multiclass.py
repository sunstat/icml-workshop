import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
# making class names shorter
class_names = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:]) for x in newsgroups_train.target_names]
class_names[3] = 'pc.hardware'
class_names[4] = 'mac.hardware'
print(class_names)


vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=.01)
nb.fit(train_vectors, newsgroups_train.target)

pred = nb.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='weighted')

from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, nb)

print(c.predict_proba([newsgroups_test.data[0]]).round(3))

from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)


def explain(num_iterations):

    idx = 1340
    features = dict()
    for iter in range(num_iterations):
        exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6, labels=[0, 17])
        for feature, weight in exp.as_list(label = 0):
            if feature not in features:
                features[feature] = 0
            features[feature] += 1
    print(features)

res = {'Caused': 50, 'Rice': 50, 'Genocide': 50, 'owlnet': 46, 'Semitic': 21, 'scri': 44, 'certainty': 37, 'Theism': 2}
vals = [1, 1, 1, 46.0/50.0, 21.0/50.0, 44.0/50.0, 37.0/50.0, 2.0/50.0]
import matplotlib.pyplot as plt
bars = plt.bar(list(res.keys()), res.values())
plt.ylabel('Freq')
plt.xlabel('Word')
for bar_idx in range(len(bars)):
    bars[bar_idx].set_color(((1-vals[bar_idx])*0.8, (1-vals[bar_idx])*0.8, (1-vals[bar_idx])*0.8))
plt.show()

if __name__ == "__main__":

    explain(10)
