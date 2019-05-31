import numpy as np
import lime
import simulation as simu
from sklearn import metrics
import matplotlib.pyplot as plt

def explain_tabular(data, explainer, black_model, num_features, num_samples, radius=1.0, verbose = 1):
    '''
    :data: the data point we would like to explain
    :explainer: a lime tabular explainer
    :black_model:
    :num_features: number of features included in the explanation
    :num_samples: number of samples to generate used to train local weighted regression model
    '''
    exp = explainer.explain_instance(data, black_model.predict_proba, num_features = num_features, num_samples = num_samples, radius=radius)

    if verbose:
        print(exp.as_list()) 
    
    return exp

def plot_features_bar(features_dict, active_features, model, label, num_iters, num_samples, radius=1.0):

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

    active_dict_err = dict()
    other_dict_err = dict()
    for key in active_dict:
        active_dict[key] /= (1.0 * num_iters)
        active_dict_err[key] = (active_dict[key] * (1 - active_dict[key])) / num_iters
    for key in other_dict:
        other_dict[key] /= (1.0 * num_iters)
        other_dict_err[key] = (other_dict[key] * (1 - other_dict[key])) / num_iters   
    #error_kw = {'capsize': 2.0, 'ecolor': 'blue'}
    plt.bar(active_dict.keys(), active_dict.values(), color = 'red', yerr=active_dict_err.values(), )
    plt.bar(other_dict.keys(), other_dict.values(), color = 'black', yerr=other_dict_err.values())
    plt.ylabel('Probability')
    plt.xlabel('Features')
    plt.legend(feature_class)
    plt.savefig('lime_tabular_8_features/bar_plots/probability/radius_{}_final/label{}_{}_local_sample_{}_sample_radius_{}.png'.format(radius, label, model, num_samples, radius))
    plt.close()
    #plt.show()  

def do_explanation(data, model, fs_mode = 'highest_weights', num_samples=1000, iteration=100, radius=1.0):

    train_X, train_y, test_X, test_y,  black_model, coefs, dtree = simu.test_generate_data(model=model)
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=train_X, 
                                                       training_labels=train_y, 
                                                       feature_selection=fs_mode, 
                                                       discretize_continuous=False,
                                                       verbose=False, 
                                                       sample_around_instance=True, 
                                                       mode='classification', 
                                                       categorical_features=None, 
                                                       categorical_names=None)
    pred_y = black_model.predict(test_X)
    print("Accuracy:", metrics.accuracy_score(test_y, pred_y))
    res = []
    for iter in range(iteration):
        curr_explanation = explain_tabular(data, explainer, black_model, num_features=3, num_samples=num_samples, radius=radius).as_list()
        print(curr_explanation)
        res.append(curr_explanation)
    return res, coefs, dtree

def extract_feature_from_explanation(exp):
    
    feature_list = []
    for e in exp:
        feature_str = e[0]
        if len(feature_str) > 1:
            splitted = feature_str.split(' ')
            if len(splitted) == 3:
                feature_list.append(int(splitted[0]))
            else:
                feature_list.append(int(splitted[2]))
        else:
            feature_list.append(int(feature_str))
    
    return feature_list

def get_active_features(data, dtree, coefs):
   
    index = dtree.get_label(data)
    features_set = set()
    for i in range(len(coefs[index])):
        if 1 == coefs[index][i]:
            features_set.add(i)
    return index, features_set

if __name__ == "__main__":
    test_points  = [[-0.2, -0.5, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                    [-0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                    [0.2, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                    [0.6, 0.6, -0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                    [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                    [0.8, 0.9, -0.8, 0.8, 0.8, 0.8, 0.8, 0.8]]
    test_points  = [[-0.8, 0.2, 0.59, 0.5, 0.5, 0.5, 0.5, 0.5],
                    [-0.8, 0.8, 0.03, -0.8, 0.5, 0.5, 0.5, 0.5],
                    [0.2, 0.6, 0.8, -0.5, -0.32, 0.5, 0.5, 0.5],
                    [0.6, 0.6, -0.8, 0.5, -0.5, 0.02, 0.5, 0.5],
                    [0.8, 0.9, -0.8, 0.5, 0.5, -0.5, 0.02, 0.5],
                    [0.8, 0.8, 0.8, 0.5, 0.5, 0.5, -0.5, 0.02]]
    num_samples=100
    for radius in [0.1, 1.0]:
        for point in test_points:
            data = np.array(point)
            for model in ['RF', 'GBT']:
                explanation, coefs, dtree = do_explanation(data, model=model, fs_mode='lasso_path', num_samples=num_samples, radius=radius)
                label, active_features = get_active_features(data, dtree, coefs)

                features_count = dict()
                for feature in range(0, 8):
                    features_count[feature] = 0
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
                plot_features_bar(features_count, active_features, model=model, label=label, num_iters=100, num_samples=num_samples, radius=radius)

