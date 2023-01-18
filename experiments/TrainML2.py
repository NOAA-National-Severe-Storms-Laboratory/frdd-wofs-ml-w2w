# Always keep this import at the top of your script. It is uses the Intel extension 
# for scikit-learn, which improves the training speed of machine learning algorithms
# in scikit-learn. 

# We add the github package to our system path so we can import python scripts for that repo. 
import sys
import argparse
sys.path.append('/home/samuel.varga/projects/2to6_hr_severe_wx/')
sys.path.append('/home/samuel.varga/python_packages/ml_workflow/')
from main.io import load_ml_data
from ml_workflow.calibrated_pipeline_hyperopt_cv import CalibratedPipelineHyperOptCV

# Import packages 
import pandas as pd
import numpy as np
import sklearn
from os.path import join
from sklearn.linear_model import LogisticRegression
from hyperopt import hp

#######
#Usage#
#######

#Change to directory of script before running
#nohup python TrainML.py -o -env -hn HAZARDNAME -ts TRAININGSCALE -hs HAZARDSCALE


#Command line input
#Hazard for target (tornado, hail, wind, all)
#Resolution for target hazard (9, 15, 36)
#Resolution for input (9, 27, 45, all)

parser=argparse.ArgumentParser()
parser.add_argument('-o', '--original', help="Original Variables", action='store_true')
parser.add_argument('-ts', '--training_scale', help="Scale of Training variables (9,27,45)")
parser.add_argument('-hs','--hazard_scale', help="Scale of Target Variables (9,15,36)")
parser.add_argument('-hn', '--hazard_name', help="Target: hail, wind, tornado")
parser.add_argument('-env','--environmental', help="Drop all intrastorm variables", action='store_true')
parser.add_argument('-is' ,'--intrastorm', help="Drop all environmental variables", action='store_true')

args=parser.parse_args()

# Configuration variables (You'll need to change based on where you store your data)
FRAMEWORK='POTVIN'
TIMESCALE='2to6'
base_path = f'/work/samuel.varga/data/{TIMESCALE}_hr_severe_wx/{FRAMEWORK}'

###########
#Data Prep#
###########

#Load Data
if args.hazard_name == 'all':    
    target_col=f'wind_severe__{args.hazard_scale}km'
    X,y,metadata = load_ml_data(base_path=base_path, 
                            mode='train', 
                            target_col=target_col,
                           FRAMEWORK=FRAMEWORK,
                           TIMESCALE=TIMESCALE)
    print(len(y[y>0]))
    for hazard in ['hail','tornado']:
        target_col='{}_severe__{}km'.format(hazard, args.hazard_scale)
        SPAM, y1, SPAM  = load_ml_data(base_path=base_path, mode='train', target_col=target_col, FRAMEWORK=FRAMEWORK, TIMESCALE=TIMESCALE) 
        y +=y1
        print(len(y[y>0]))
       
    y[y > 0] = 1
    
    
else:
    target_col='{}_severe__{}km'.format(args.hazard_name, args.hazard_scale)

    
    
    X,y,metadata = load_ml_data(base_path=base_path, 
                            mode='train', 
                            target_col=target_col,
                           FRAMEWORK=FRAMEWORK,
                           TIMESCALE=TIMESCALE)


X=X.drop(['NX','NY'], axis=1)

#Select Desired Input Scales:
if args.training_scale:
    X=X[[col for col in X.columns if '{}km'.format(args.training_scale) in col]] #Removes all columns except for those with correct scale, while keeping the same number of rows
    ts_suff=str(args.training_scale)+'km'
else:
    ts_suff='all'

#All input scales are used if -hs is not used    


#Use original 90th percentile and drop all new vars if -o is used
vardic={ 'ENS_VARS':  ['uh_2to5_instant',
                            'uh_0to2_instant',
                            'wz_0to2_instant',
                            'comp_dz',
                            'ws_80',
                            'hailcast',
                            'w_up',
                            'okubo_weiss',
                    ]}

if args.original:
    print("Using Original Variables- Dropping IQR, 2nd lowest, 2nd highest, and intrastorm mean")
    X=X[[col for col in X.columns if 'IQR' not in col]]
    X=X[[col for col in X.columns if '2nd' not in col]]
    X=X[[col for col in X.columns if '16th' not in col]]
    #Mean of intrastorm vars
   
    badthings=np.array([])
    for strmvar in vardic['ENS_VARS']:
        badthings=np.append(badthings, [col for col in X.columns if 'mean' in col and strmvar in col] )

    X=X.drop(badthings, axis=1)
    
else:
    print("Using new variables- dropping old 90th percentile")
    X=X[[col for col in X.columns if '90th' not in col]] #Keeps all columns except the old 90th %ile

        

#Drop all intrastorm vars if -env is used, drop all environmental vars if -is is used
if args.environmental:
    print("Dropping all intrastorm variables")
    stormcols=np.array([])
    for strmvar in vardic['ENS_VARS']:
        stormcols=np.append(stormcols, [col for col in X.columns if strmvar in col]) #Every column name that has a storm var
    X=X.drop(stormcols, axis=1) #Drops all IS columns    
elif args.intrastorm:
    print("Dropping all env vars")
    stormcols=np.array([])
    for strmvar in vardic['ENS_VARS']:
        stormcols=np.append(stormcols, [col for col in X.columns if strmvar in col])
    X=X[stormcols]
                
        
#Debugging    
print(args)
print(target_col)    
#print(list(X.columns))    
print(X.shape)    
print(ts_suff)
#exit()


#################
#Training models#
#################

scaler = 'standard'
resample = 'under'


#names=['logistic','random','hist','ADAM']
names=['ADAM']

for name in names:
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
        base_estimator=sklearn.ensemble.HistGradientBoostingClassifier(random_state=42, max_iter=150)
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
        
        
    elif name=='ADAM': #Add a conditional to reduce the number of max iterations for this
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
        max_iter=25
    else:
        max_iter=50


    train_dates = metadata['Run Date'].apply(str)

    clf = CalibratedPipelineHyperOptCV(base_estimator=base_estimator, 
                                       param_grid=param_grid, 
                                       scaler=scaler, 
                                       resample=resample, max_iter=max_iter, 
                                       cv_kwargs = {'dates': train_dates, 'n_splits': 5, 'valid_size' : 20}, 
                                       hyperopt='tpe')

    clf.fit(X, y)

    if args.original or args.environmental:
        suff=''
        if args.original:
            suff+='control_'
        if args.environmental:
            suff+='env'
        save_name = f'Varga_{ts_suff}_{name}_{args.hazard_name}_{args.hazard_scale}km_{suff}.joblib'
    else:
        save_name = f'Varga_{ts_suff}_{name}_{args.hazard_name}_{args.hazard_scale}km.joblib'
    #print(save_name)
    clf.save(save_name) 
