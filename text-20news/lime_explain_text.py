import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics

from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']

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

print(c.predict_proba([newsgroups_test.data[0]]))


from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)

'''
feats = dict()
for idx in range(20):
    exp = explainer.explain_instance(newsgroups_test.data[idx], 
                                 c.predict_proba, 
                                 num_features=6, 
                                 labels=[0, 1])
    for keyword in exp.as_list(0):
        if not keyword[0] in feats:
            feats[keyword[0]] = 0
        feats[keyword[0]] += 1
    print('Document id: %d' % idx)
    print('Probability(christian) =', c.predict_proba([newsgroups_test.data[idx]])[0,1])
    print('True class: %s' % class_names[newsgroups_test.target[idx]])
print(feats)
'''
'''
feats = dict()
idx = 10
for iteration in range(100):
    exp = explainer.explain_instance(newsgroups_test.data[idx], 
                                 c.predict_proba, 
                                 num_features=6, 
                                 labels=[0, 1])
    for keyword in exp.as_list(0):
        if not keyword[0] in feats:
            feats[keyword[0]] = 0
        feats[keyword[0]] += 1
    print('Document id: %d' % idx)
    print('Probability(christian) =', c.predict_proba([newsgroups_test.data[idx]])[0,1])
    print('True class: %s' % class_names[newsgroups_test.target[idx]])
print(feats)
'''
idx_0_19_explanations = {'article': 7, 'au': 1, 'guy': 1, 'wrote': 2, 'deleted': 1,
'edu': 3, 'morality': 1, 'clh': 3, 'don': 1, 'to': 3, 'than': 2, 'atheist': 1, 'Keith': 1, 'Re': 8, 'writes': 5, 'California': 1, 'atheists': 3, 'Schneider': 1, 'Christians': 4, 'athos': 3, 'rutgers': 4, 'the': 5, 'Christ': 5, 'Host': 3, 'Posting': 3, 'NNTP': 3, 'God': 7, 'geneva': 1, '1993': 2, 'logical': 1, 'In': 6, 'sgi': 1, 'definition': 1, 'de': 1, 'thing': 1, 'Rushdie': 1, 'Boswell': 1, 'church': 2, 'of': 3, 'solntze': 1, 'wouldn': 1, 'higher': 1, 'absurd': 1, 'try': 1, 'murder': 1, 'and': 2, 'position': 1, 'Bill': 1, 'sin': 1,
'love': 1, 'He': 1, 'more': 1, 'moral': 1, 'in': 1, 'so': 1}

idx_10_explanations = {'rutgers': 100, 'Christ': 100, '1993': 100, 'athos': 97, 
                       'writes': 88, 'article': 100, 'in': 2, 'to': 1, 
                       'thing': 2, 'edu': 10}

electronics_crypt_1 = {'crypto': 100, 'netcom': 100, 'Sternlight': 100, 
                       'scheme': 99, 'people': 89, 'Re': 10, 'strnlght': 95, 
                       'In': 2, 'com': 5}
color = ['r', 'r', 'r', 'black', 'black', 'black', 'r', 'black', 'black']

import matplotlib.pyplot as plt
bars = plt.bar(list(electronics_crypt_1.keys()), electronics_crypt_1.values())
plt.ylabel('Freq')
plt.xlabel('Word')
for bar_idx in range(len(bars)):
    bars[bar_idx].set_color(color[bar_idx])
plt.show()
