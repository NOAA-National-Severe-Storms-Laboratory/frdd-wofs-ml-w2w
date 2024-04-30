# Always keep this import at the top of your script. It is uses the Intel extension 
# for scikit-learn, which improves the training speed of machine learning algorithms
# in scikit-learn. 

# We add the github package to our system path so we can import python scripts for that repo. 
import sys
import argparse
#Appendages
sys.path.append('/home/samuel.varga/projects/2to6_hr_severe_wx/')
sys.path.append('/home/samuel.varga/python_packages/ml_workflow/')
sys.path.append('/home/samuel.varga/python_packages/VargaPy/')
# Import packages 
import pandas as pd
import numpy as np
import sklearn
from os.path import join
from sklearn.linear_model import LogisticRegression
from hyperopt import hp
from main.io import load_ml_data
from ml_workflow.calibrated_pipeline_hyperopt_cv import norm_aupdc_scorer, norm_csi_scorer
from ml_workflow.tuned_estimator import TunedEstimator, dates_to_groups
from VargaPy.MlUtils import All_Severe, Drop_Unwanted_Variables, Train_Ml_Parser
from sklearn.model_selection import GroupKFold
from itertools import product

#######
#Usage#
#######

#Change to directory of script before running
#nohup python RandSampTrain.py -o -env -hn HAZARDNAME -ts TRAININGSCALE -hs HAZARDSCALE
#nohup python RandSampTrain.py -hn all -hs all


#Command line input
#Hazard for target (tornado, hail, wind, all)
#Resolution for target hazard (9, 15, 36, all)
#Resolution for input (9, 27, 45, all=None)

parser=Train_Ml_Parser()
args=parser.parse_args()

# Configuration variables (You'll need to change based on where you store your data)
framework=['POTVIN']
timescale=['2to6']
Tkm=False
mod_names=['hist']
hazard_scale = [36,18,9] if args.hazard_scale == 'all' else [args.hazard_scale]
HAZARD=['wind','hail','tornado'] if args.hazard_name == ['each'] else args.hazard_name


#############################
##Tuned Estimator Arguments##
#############################

arguments_dict = {'pipeline_arguments':{#Dictionary of arguments for ml_workflow.preprocess.PreProcessingPipeline
                        'imputer':'simple', #From sklearn.impute- handles missing data- simple or iterative
                        'scaler':'standard', #From sklearn.preprocessing - scales features - standard, robust, minmax
                        'pca':None, #From sklearn.decomposition - method of PCA - None, or valid methods
                        'resample':None, #imblearn.under/over_sampling - Resamples training folds of KFCV- under, None, over 
                        'sampling_strategy':None, #Default setting
                        'numeric_features':None,
                        'categorical_features':None
 
},
             'hyperopt_arguments':{ #Dictionary of arguments for ml_workflow.hyperparameter_optimizer.HyperOptCV
                        'search_space':None, #Update
                        'optimizer':'tpe', #None or tpe
                        'max_evals':35,
                        'patience':25,
                        'scorer':norm_csi_scorer,
                        'n_jobs':None, #None
                        'cv':None, #Updated later
                         'output_fname':None
}, 
             'calibration_arguments': {#Dictionary of arguments for sklearn.calibration.CalibratedClassifierCV
                        'method':'isotonic',
                        'cv':None, #Updated later
                        'n_jobs':None, #None
                        'ensemble':False                        
} 
            
            
            }
#############
##Data Prep##
#############
#Change hazard back to full 9km

###Start Loop here
for radius, FRAMEWORK, TIMESCALE, hazard in product(hazard_scale, framework, timescale, HAZARD):
    print(f'Starting the process for:')
    print(f'{radius} {FRAMEWORK} {TIMESCALE} {hazard}')
    base_path = f'/work/samuel.varga/data/{TIMESCALE}_hr_severe_wx/sfe_prep' #{FRAMEWORK}'
    OUTPATH=f'/work/samuel.varga/projects/{TIMESCALE}_hr_severe_wx/sfe_prep/mlModels/' #{radius}km'
    #{FRAMEWORK}/mlModels/{radius}km'
    
    
    #Load Data
    if hazard == 'all':    
        target_col=f'tornado_severe__{radius}km' #Used to grab the correct baseline
        X,y,metadata = All_Severe(base_path, mode='train',
                                  target_scale=radius,
                                  FRAMEWORK=FRAMEWORK,
                                  TIMESCALE=TIMESCALE, SigSevere=args.SigSevere, appendUH=False, Three_km=Tkm, full_9km=True)
    else:
        target_col='{}_severe__{}km'.format(hazard, radius)
        print(target_col)
        X,y,metadata = load_ml_data(base_path=base_path, 
                                mode='train', 
                                target_col=target_col,
                               FRAMEWORK=FRAMEWORK,
                               TIMESCALE=TIMESCALE, Three_km=Tkm, full_9km=True)

  
    X, ts_suff, var_suff = Drop_Unwanted_Variables(X, original=args.original, training_scale=args.training_scale, intrastormOnly=args.intrastorm, envOnly=args.environmental)
    

    
    #Debugging    
    print(args)
    #print(target_col)    
    #print(list(X.columns))    
    print(X.shape)    
    print(ts_suff)
    #exit()

    
    
###################
##Training models##
###################


         
    for n in [0]:
        print(f'Starting {n} process for {radius}km ')

        train_dates=metadata['Run Date']
        groups=dates_to_groups(train_dates, n_splits=5)
        cv=list(GroupKFold(n_splits=5).split(X, y, groups))
        arguments_dict['hyperopt_arguments']['cv'], arguments_dict['calibration_arguments']['cv']= cv, cv          



        for name in mod_names:#random, hist, logistic, ADAM
                print(f'Starting Learning for {name}')
                if name=='logistic':
                    base_estimator = LogisticRegression(solver='saga', penalty='elasticnet', max_iter=300, random_state=42, l1_ratio=0.001)
                    #Param grid for LogReg
                    param_grid = {
                                'l1_ratio': [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.6, 0.8, 1.0],
                                'C': [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.62, 0.75, 0.87, 1.0],
                            }
                elif name=='random':
                    base_estimator=sklearn.ensemble.RandomForestClassifier(random_state=42, n_estimators=200, max_depth=15, max_features='sqrt', min_samples_leaf=20)
                    #Param Grid for RF
                    param_grid = {
                               'n_estimators' : [10, 25, 50, 100,150,300,400,500], 
                               'max_depth' : [4, 6,8,10,15,20],
                               'max_features' : [4,6,8,10,15,20,25,30],
                               'min_samples_split' : [4,6,8,10,15,20,25,50],
                               'min_samples_leaf' : [4,6,8,10,15,20,25,50],
                            }
                elif name=='hist':
                    base_estimator=sklearn.ensemble.HistGradientBoostingClassifier(random_state=42, learning_rate=0.001, max_leaf_nodes=20, max_depth=15, min_samples_leaf=20, l2_regularization=0.001, max_bins=31)
                    #Check performance of different loss functions?
                    #Param Grid for HGB
                    param_grid= {
                    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
                    'max_leaf_nodes': [5, 10, 20, 30, 40, 50],
                    'max_depth': [4, 6, 8, 10],
                    'min_samples_leaf': [5,10,15,20,30, 40, 50],
                    'l2_regularization': [0.001, 0.01, 0.1], 
                    'max_bins': [15, 31, 63, 127]

                            }


                elif name=='rand-entropy':
                    base_estimator=sklearn.ensemble.RandomForestClassifier(random_state=42, n_estimators=200, criterion='entropy', max_depth=15, max_features='sqrt', min_samples_leaf=20)
                    param_grid = None

               
                arguments_dict['hyperopt_arguments']['search_space']=param_grid
                arguments_dict['hyperopt_arguments']['output_fname']=f'/home/samuel.varga/.hpopt/{ts_suff}_{name}_{hazard}_{radius}.feather'
                t_e = TunedEstimator(estimator=base_estimator,
                                     pipeline_kwargs=arguments_dict['pipeline_arguments'],
                                     hyperopt_kwargs=None if name=='rand-entropy' else arguments_dict['hyperopt_arguments'],
                                     calibration_cv_kwargs=arguments_dict['calibration_arguments'])


                t_e.fit(X, y, groups) 

                save_name = f'sfe_{ts_suff}_{name}_{hazard}_{radius}km_{"SigSev" if args.SigSevere else "Sev"}_{var_suff}_{n}{"_DBRS" if Tkm else ""}.joblib'
                print(join(OUTPATH,save_name))
                t_e.save(join(OUTPATH, save_name)) 
