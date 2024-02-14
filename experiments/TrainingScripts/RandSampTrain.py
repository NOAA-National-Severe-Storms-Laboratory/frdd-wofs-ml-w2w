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
from VargaPy.MlUtils import All_Severe, Simple_Random_Subsample, Drop_Unwanted_Variables, Train_Ml_Parser
from sklearn.model_selection import GroupKFold

#######
#Usage#
#######

#Change to directory of script before running
#nohup python RandSampTrain.py -o -env -hn HAZARDNAME -ts TRAININGSCALE -hs HAZARDSCALE
#nohup python RandSampTrain.py -hn all -hs all


#Command line input
#Hazard for target (tornado, hail, wind, all)
#Resolution for target hazard (9, 15, 36)
#Resolution for input (9, 27, 45, all)

parser=Train_Ml_Parser()
args=parser.parse_args()

# Configuration variables (You'll need to change based on where you store your data)
FRAMEWORK='POTVIN'
TIMESCALE='0to3'
base_path = f'/work/samuel.varga/data/{TIMESCALE}_hr_severe_wx/{FRAMEWORK}'


#############################
##Tuned Estimator Arguments##
#############################

arguments_dict = {'pipeline_arguments':{#Dictionary of arguments for ml_workflow.preprocess.PreProcessingPipeline
                        'imputer':'simple', #From sklearn.impute- handles missing data- simple or iterative
                        'scaler':'standard', #From sklearn.preprocessing - scales features - standard, robust, minmax
                        'pca':None, #From sklearn.decomposition - method of PCA - None, or valid methods
                        'resample':None, #imblearn.under/over_sampling - Resamples training folds of KFCV- under, None, over 
                        'sampling_strategy':'auto', #Default setting
                        'numeric_features':None,
                        'categorical_features':None
 
},
             'hyperopt_arguments':{ #Dictionary of arguments for ml_workflow.hyperparameter_optimizer.HyperOptCV
                        'search_space':None, #Update
                        'optimizer':'tpe', #atpe or tpe
                        'max_evals':50,
                        'patience':10,
                        'scorer':norm_csi_scorer,
                        'n_jobs':1,
                        'cv':None #Update
}, 
             'calibration_arguments': {#Dictionary of arguments for sklearn.calibration.CalibratedClassifierCV
                        'method':'isotonic',
                        'cv':None, #Update
                        'n_jobs':None,
                        'ensemble':False                        
} 
            
            
            }
#############
##Data Prep##
#############
if args.hazard_scale =='all':
    hazard_scale=[36, 18, 9]
else:
    hazard_scale=[args.hazard_scale]

###Start Loop here
for radius in hazard_scale:
    print(f'Starting the process for {radius}km')
    
    OUTPATH=f'/work/samuel.varga/projects/{TIMESCALE}_hr_severe_wx/{FRAMEWORK}/mlModels/{radius}km'
    Seed=np.random.RandomState(42)    
    
    #Load Data
    if args.hazard_name == 'all':    
        target_col=f'tornado_severe__{radius}km' #Used to grab the correct baseline
        X,y,metadata = All_Severe(base_path, mode='train',
                                  target_scale=radius,
                                  FRAMEWORK=FRAMEWORK,
                                  TIMESCALE=TIMESCALE, Big=True)
    else:
        target_col='{}_severe__{}km'.format(args.hazard_name, radius)
        X,y,metadata = load_ml_data(base_path=base_path, 
                                mode='train', 
                                target_col=target_col,
                               FRAMEWORK=FRAMEWORK,
                               TIMESCALE=TIMESCALE)

  
    X, ts_suff = Drop_Unwanted_Variables(X, original=args.original, training_scale=args.training_scale, intrastormOnly=args.intrastorm, envOnly=args.environmental)
    


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


    #Second Loop for percentages
    for p in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1]:
        for n in [0,1,2,3,4]:
            print(f'Starting {n} process for {radius}km {p*100}%')

            ##Get the data subset
            X_sub, y_sub, meta_sub = Simple_Random_Subsample(X, y, metadata, p, Seed)
            print(f'input shape for {radius}km {p*100}%: {np.shape(X_sub)}')

            train_dates=meta_sub['Run Date']
            groups=dates_to_groups(train_dates, n_splits=5)
            cv=list(GroupKFold(n_splits=5).split(X_sub, y_sub, groups))
            arguments_dict['hyperopt_arguments']['cv'], arguments_dict['calibration_arguments']['cv']= cv, cv



            for name in ['hist']:
                if name=='logistic':
                    base_estimator = LogisticRegression(solver='saga', penalty='elasticnet', max_iter=300, random_state=42)
                    #Param grid for LogReg
                    param_grid = {
                                'l1_ratio': hp.choice('l1_ratio', [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.6, 0.8, 1.0]),
                                'C': hp.choice('C', [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.62, 0.75, 0.87, 1.0]),
                            }
                elif name=='random':
                    base_estimator=sklearn.ensemble.RandomForestClassifier(random_state=42)
                    #Param Grid for RF
                    param_grid = {
                               'n_estimators' : hp.choice('n_estimators',[10, 25, 50, 100,150,300,400,500]), 
                               'max_depth' : hp.choice('max_depth',[4, 6,8,10,15,20]),
                               'max_features' : hp.choice('max_features',[4,6,8,10,15,20,25,30]),
                               'min_samples_split' : hp.choice('min_samples_split',[4,6,8,10,15,20,25,50]),
                               'min_samples_leaf' : hp.choice('min_samples_leaf',[4,6,8,10,15,20,25,50]),
                            }
                elif name=='hist':
                    base_estimator=sklearn.ensemble.HistGradientBoostingClassifier(random_state=42, max_iter=150, l2_regularization=0.01, learning_rate=0.01, max_bins=15, max_depth=8, max_leaf_nodes=30, min_samples_leaf=20)
                    #Check performance of different loss functions?
                    #Param Grid for HGB
                    param_grid= {
                    'learning_rate': hp.choice('learning_rate',[0.0001, 0.001, 0.01, 0.1]),
                    'max_leaf_nodes': hp.choice('max_leaf_nodes',[5, 10, 20, 30, 40, 50]),
                    'max_depth': hp.choice('max_depth', [4, 6, 8, 10]),
                    'min_samples_leaf': hp.choice('min_samples_leaf',[5,10,15,20,30, 40, 50]),
                    'l2_regularization': hp.choice('l2_regularization',[0.001, 0.01, 0.1]), #This one causes problems
                    'max_bins': hp.choice('max_bins',[15, 31, 63, 127])

                            }


                elif name=='ADAM':
                    base_estimator=sklearn.ensemble.RandomForestClassifier(random_state=42)
                    param_grid = {
                           'n_estimators' : hp.choice('n_estimators',[200]), 
                           'criterion' : hp.choice('criterion',['entropy']),
                            'max_depth' : hp.choice('max_depth',[15]),
                           'max_features' : hp.choice('max_features',["sqrt"]),
                           #'min_samples_split' : hp.choice('min_samples_split',[4,6,8,10,15,20,25,50]),
                           'min_samples_leaf' : hp.choice('min_samples_leaf',[20])
                        }


                if name=='ADAM':
                    max_iter=50 #25
                else:
                    max_iter=50



                arguments_dict['hyperopt_arguments']['search_space']=param_grid
                t_e = TunedEstimator(estimator=base_estimator,
                                     pipeline_kwargs=arguments_dict['pipeline_arguments'],
                                     hyperopt_kwargs=None,
                                     calibration_cv_kwargs=arguments_dict['calibration_arguments'])


                t_e.fit(X_sub, y_sub, groups)

                if args.original or args.environmental:
                    suff=''
                    if args.original:
                        suff+='control_'
                    if args.environmental:
                        suff+='env'
                    save_name = f'Varga_{ts_suff}_{name}_{args.hazard_name}_{args.hazard_scale}km_{suff}.joblib'
                else:
                    save_name = f'Varga_{ts_suff}_{name}_{args.hazard_name}_{radius}km_{int(p*100)}_{n}.joblib'
                print(save_name)
                t_e.save(join(OUTPATH, save_name)) 
