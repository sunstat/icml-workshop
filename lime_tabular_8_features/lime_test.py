import numpy as np
import lime
import simulation as simu
from sklearn import metrics
import matplotlib.pyplot as plt

def explain_tabular(data, explainer, black_model, num_features, num_samples, verbose = 1):
    '''
    :data: the data point we would like to explain
    :explainer: a lime tabular explainer
    :black_model:
    :num_features: number of features included in the explanation
    :num_samples: number of samples to generate used to train local weighted regression model
    '''
    exp = explainer.explain_instance(data, black_model.predict_proba, num_features = num_features, num_samples = num_samples)

    if verbose:
        print(exp.as_list()) 
    
    return exp

def plot_features_bar(features_dict, active_features):

    color_list = []
    active_dict = dict()
    other_dict = dict()
    feature_class = ['Active Features', 'Inactive Features']
    for features_name in features_dict.keys():
        if features_name in active_features:
            color_list.append('red')
            active_dict[features_name] = features_dict[features_name]
        else:
            color_list.append('black')
            other_dict[features_name] = features_dict[features_name]
        
    plt.bar(active_dict.keys(), active_dict.values(), color = 'red')
    plt.bar(other_dict.keys(), other_dict.values(), color = 'black')
    plt.ylabel('Freq')
    plt.xlabel('Features')
    plt.legend(feature_class, loc=1)
    plt.show()  

def do_explanation(data, fs_mode = 'highest_weights', iteration = 100):

    train_X, train_y, test_X, test_y,  black_model, coefs, dtree = simu.test_generate_data()
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=train_X, training_labels=train_y, feature_selection=fs_mode,
                                                       verbose=False, mode='classification', categorical_features=None, categorical_names=None)
    pred_y = black_model.predict(test_X)
    print("Accuracy:", metrics.accuracy_score(test_y, pred_y))
    res = []
    for iter in range(iteration):
        curr_explanation = explain_tabular(data, explainer, black_model, num_features = 3, num_samples = 100).as_list()
        print(curr_explanation)
        res.append(curr_explanation)
    return res, coefs, dtree

def extract_feature_from_explanation(exp):
    
    feature_list = []
    for e in exp:
        feature_str = e[0]
        splitted = feature_str.split(' ')
        if len(splitted) == 3:
            feature_list.append(int(splitted[0]))
        else:
            feature_list.append(int(splitted[2]))
    
    return feature_list

def get_active_features(data, dtree, coefs):
   
    index = dtree.get_label(data)
    features_set = set()
    for i in range(len(coefs[index])):
        if 1 == coefs[index][i]:
            features_set.add(i)
    return index, features_set

if __name__ == "__main__":
    data = np.array([0.2, 0.9, -0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    explanation, coefs, dtree = do_explanation(data, fs_mode = 'lasso_path')
    label, active_features = get_active_features(data, dtree, coefs)
    features_count = dict()
    features_list = []
    for exp in explanation:
        features = extract_feature_from_explanation(exp)
        for feature in features:
            features_list.append(feature)
            if feature not in features_count:
                features_count[feature] = 0
            features_count[feature] += 1
    print(features_count)
    print('label: ', label)
    plot_features_bar(features_count, active_features)

