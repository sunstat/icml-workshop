import numpy as np
import lime
import simulation as simu
from sklearn import metrics
import matplotlib.pyplot as plt

def explain_tabular(data, explainer, black_model, num_features = 2, num_samples = 100, verbose = 1):
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

def do_explanation(data, model, iteration=100):

    train_X, train_y, test_X, test_y,  black_model = simu.test_generate_data(model=model)
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=train_X,  
                                                       training_labels=train_y, 
                                                       feature_selection='lasso_path',
                                                       discretize_continuous=False,
                                                       sample_around_instance=True,
                                                       verbose=False, 
                                                       mode='classification', 
                                                       categorical_features=None, 
                                                       categorical_names=None)
    pred_y = black_model.predict(test_X)
    print("Accuracy:", metrics.accuracy_score(test_y, pred_y))
    res = []

    for iter in range(iteration):
        curr_explanation = explain_tabular(data, explainer, black_model, num_features = 2, num_samples = 100, verbose = 0).as_list()
        print(curr_explanation)
        res.append(curr_explanation)
    return res

def plot_features_bar(features_dict, active_features, label, model, num_iters=100):

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
    
    for key in active_dict:
        active_dict[key] /= (1.0 * num_iters)
    for key in other_dict:
        other_dict[key] /= (1.0 * num_iters)   

    plt.bar(active_dict.keys(), active_dict.values(), color = 'red')
    plt.bar(other_dict.keys(), other_dict.values(), color = 'black')
    plt.ylabel('Probability')
    plt.xlabel('Features')
    plt.xticks([0, 1, 2, 3])
    plt.legend(feature_class, loc=1)
    plt.savefig('lime_tabular_4_features/figures/probability_bar/label{}_{}.png'.format(label, model))
    plt.close()
    #plt.show()  

def determine_region(x):

    # x0(x axis), x1(y axis), x2, x3
    if x[1]>0:
        if x[0]>0:
            return 0
        else:
            return 1
    else:
        if x[0]>0:
            return 3
        else:
            return 2

if __name__ == "__main__":
    coefs = [[1,1,0,0], [0,1,1,0], [0,0,1,1], [1,0,0,1]]
    data = np.array([0.5, -0.5, 0.5, 0.5])
    label = determine_region(data)
    model='RF'
    explanation = do_explanation(data, 0 if 'RF'==model else 1)
    features_list = []
    features_count = dict()
    for exp in explanation:
        features = extract_feature_from_explanation(exp)
        for feature in features:
            features_list.append(feature)
            if feature not in features_count:
                features_count[feature] = 0
            features_count[feature] += 1
    print(features_count)
    active_features = set([i for i, value in enumerate(coefs[label]) if value])
    plot_features_bar(features_count, active_features, label=label, model=model, num_iters=100)