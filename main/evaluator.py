import sys
sys.path.append('/home/monte.flora/python_packages/ml_workflow/')

import numpy as np
from mlxtend.evaluate import permutation_test
from ml_workflow import DateBasedCV
from ml_workflow.ml_methods import norm_aupdc, norm_csi, brier_skill_score
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression


# A function that takes a model, model input, and the true labels 
def scorer(model, X, y, **kwargs ):
    known_skew = kwargs.get('known_skew', np.mean(y))
    predictions = model.predict(X)
    #return norm_aupdc(y, predictions, known_skew=known_skew)
    #return roc_auc_score(y, predictions)
    return brier_skill_score(y, predictions)

# A cross-validation function. 
def baseline_cv_scorer(X, y, dates):
    """
    Compute the cross-validation (N=5) scores for the baseline predictions. 
    
    Parameters
    --------------------
        X : np.array or dataframe
            The raw baseline predictions 
        
        y : np.array of 0 and 1s 
            The target values 
        
        dates : list 
            The date per example. Used for the cross-validation splitting. 
        
    
    Returns
    ---------------------
        cv_scores : array of shape (5,)
            The brier skill score per cross-validation fold. 
    
    """
    known_skew = np.mean(y)
    
    cv = DateBasedCV(n_splits=5, dates=dates, y=y, valid_size=0.3)
    cv_scores = []
    for train_inds, test_inds in cv.split(X):
        clf = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        clf.fit(X[train_inds], y[train_inds])
        cv_scores.append(scorer(clf, X[test_inds], y[test_inds], known_skew=known_skew))
    
    return cv_scores

# Evaluating the baseline model. 
def evaluate_baseline(df, hazard, target_col):
    baseline = {'tornado' : 'uh_2to5_instant', 
            'hail' : 'hailcast',
            'wind' : 'ws_80'
           }
    nmep_scales = [9, 27, 45]
    
    #[baseline[hazard]]__nmep_>[rng]_[9|27|45]km
    dates = df['Run Date']
    
    bl_vars = [f for f in list(X.columns) if baseline[hazard] in f]
    
    for scales in nmep_scales:
        vs = [v for v in bl_vars if f'{scale}km' in v]
        #rng = [float(v.split('_')[2].replace('>','')) for v in var_set_subset]
        for v in vs:
            X = df[[v]]
            y = df[target_col]
            
            scores = cv_scorer(X,y,dates)
            
            ymean = np.mean(scores, axis=-1)   
            ax.plot(rng, ymean, label = fr'NMEP:{n}, TAR:{k}', color=colors[i])   

            
def stat_testing(new_score, baseline_score):
    """
    Compute a p-value between two sets using permutation testing 
    to determined statistical significance. In this case,
    assess whether the ML performance is greater than the baseline.
    """
    p_value = permutation_test(new_score,
                              baseline_score,
                             'x_mean != y_mean',
                              method='approximate',
                               num_rounds=1000,
                               seed=0)
    return p_value