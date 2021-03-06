import numpy as np
import pickle
import os
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sklearn.model_selection
from sklearn import metrics



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

        



def generate_data(sample_size, feature_size, coefs):

    '''
    :param sample_size:
    :param feature_size:
    :param coefs: list_of_list
    :param blackbox_model:
    :return:
    '''


    X = np.random.uniform(-1, 1, sample_size * feature_size).reshape(sample_size, feature_size)
    y = np.array([])
    for x in X:
        '''
        determine y using determine region
        '''
        #print('x:', x)
        #print('region:', determine_region(x))
        #print('coefs:', coefs[determine_region(x)])
        #print sum(x*coefs[determine_region(x)])
        #print bool(sum(x*coefs[determine_region(x)])>=0)
        y = np.append(y,int(sum(x*coefs[determine_region(x)])>=0))
        #print y;

    train_X,  test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, y, train_size=0.80, test_size=0.20)

    return train_X, train_y, test_X, test_y





def fit_blackmodel(train_X, train_y, test_X, test_y, blackmodel):

    '''
    :param train_X:
    :param train_y:
    :param test_X:
    :param test_Y:
    :param blackbox_model: 0 if RandomForest 1 if gradient boosting
    :return:

    '''

    if blackmodel == 0: 
        rf = RandomForestClassifier()
        rf.fit(train_X, train_y)    
        pred_y = rf.predict(test_X)
        print("Accuracy:", metrics.accuracy_score(test_y, pred_y))
        return rf        

    elif blackmodel == 1: # gradient boosting
        gbc = GradientBoostingClassifier()
        gbc.fit(train_X, train_y)
        pred_y = gbc.predict(test_X)
        print("Accuracy:", metrics.accuracy_score(test_y, pred_y))
        return gbc       







def lime_app(train_X, train_y, test_X, test_y, black_pred_train_y, black_pred_test_y, black_model):

    '''
    :param x: random generated point
    :return: print lime result
    '''
    #print('train_X:', train_X)
    #print('train_y:', train_y)

    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=train_X, training_labels=train_y, feature_selection='auto',
                                                       verbose=True, mode='classification', categorical_features=None,categorical_names=None)

    i = 25
    #print(test_X[i])
    #print(determine_region(test_X[i]))
    exp = explainer.explain_instance(test_X[i], black_model.predict_proba, num_features=2, num_samples=100)   
    # to do: Print out the results

    exp.as_html(show_table=True)
    print(exp.as_list())
    exp.available_labels()
    exp.local_exp.keys()

    

def test_generate_data(model):

    '''
    :param blackbox_model: 0 if RandomForest 1 if gradient boosting
    :return:
    '''    

    coefs = [[1,1,0,0], [0,1,1,0], [0,0,1,1], [1,0,0,1]]
    pickle_path_model = os.path.join(os.getcwd()+'blackbox_{}_trained_results.pickle'.format('RF' if 0==model else 'GBT'))
    pickle_path_data =  os.path.join(os.getcwd()+'data.pickle')

    try:
        print('load data from pickle file')
        black_model = pickle.load(open(pickle_path_model,'rb'))
        data = pickle.load(open(pickle_path_data, 'rb'))

        train_X = data['train_X']
        train_y = data['train_y']
        test_X = data['test_X']
        test_y = data['test_y']

    except:
        print('no pickle file found, generating data instead')
        train_X, train_y, test_X, test_y = generate_data(5000, 4, coefs)
        black_model = fit_blackmodel(train_X, train_y, test_X, test_y,model)
        f = open(pickle_path_model ,'wb')
        pickle.dump(black_model,f)

        f.close()
        f = open(pickle_path_data,'wb')
        pickle.dump({'train_X':train_X, 'train_y':train_y,'test_X':test_X,'test_y':test_y, 'black_pred_train_y': black_model.predict(train_X), 'black_pred_test_y': black_model.predict(test_X)},f)
        f.close()

    
    return train_X, train_y, test_X, test_y, black_model
    #x_random = np.random.uniform(-1, 1, 1*4).reshape(1, 4)

    #lime_app(train_X, train_y, test_X, test_y, black_model.predict(train_X), black_model.predict(test_X), black_model)

    #y = coefs[determine_region(x_random)];


if __name__ == "__main__":

    test_generate_data()