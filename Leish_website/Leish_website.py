#Backend of website
#Everything needed to make this code work
import pandas as pd
import numpy as np
# !pip install rdkit
import xgboost
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
import chemprop
# !pip install chemprop
# !pip install git+https://github.com/bp-kelley/descriptastorus


data=str(input('type the testing file path with / = '))
output=str(input('type the output path with / = '))
df=pd.read_csv(data)
# print(type(list(df.iloc[:,0])))


def converting_SMILES_to_input(smiles=list):
  # converts list of SMILES into a dataframe for input in the model ecfp fingerprints+ rdkit 2d descriptors which are (Z-scale normalixzed)
  ecfp_l=[]
  descriptors_list=[]
  desc_names=[]
  for i in Descriptors.descList:
    desc_names.append(i[0])
  for i in smiles:
    mol = Chem.MolFromSmiles(i) # convert SMILES to RDKit molecule
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
    ecfp_l.append(ecfp)
    desc_list=[]
    for desc, func in Descriptors.descList:
        desc_list.append(func(mol))
    descriptors_list.append(desc_list)
  descriptors_l_normalized = scaler.transform(descriptors_list)
  input_dataframe= pd.concat([pd.DataFrame(np.array(ecfp_l)),pd.DataFrame(descriptors_l_normalized,columns=desc_names)],axis=1)
  input_dataframe.columns = input_dataframe.columns.astype(str)
  return input_dataframe

print('Loadind pickle files......')
# load the object from the pickle file
with open('C:/Project/Project_3/main_model_xgb_ecfp_2d_desc.pkl', 'rb') as file:
    xgb_ecfp_2d_desc = pickle.load(file)     #loads from pickle files scaler and Trained XgB model for working pipeline
with open("C:/Project/Project_3/desc_scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
#pipeline for prediction
# create a pipeline
pipeline = Pipeline([
    ('transformer', FunctionTransformer(converting_SMILES_to_input)),
    ('classifier', xgb_ecfp_2d_desc)
])
# C:\Project\to_predict_ZINC_SMILES.csv
print('predicting for XGB classifier....')
# make a prediction using the pipeline
xgb_prediction = pipeline.predict(list(df.iloc[:,0]))
xgb_prediction_proba = pipeline.predict_proba(list(df.iloc[:,0]))#input file should be a dataframe first column SMILES
active_probablity=[]
for i in xgb_prediction_proba:
  active_probablity.append(i[1])
preds=pd.concat([df.iloc[:,0],pd.DataFrame(xgb_prediction,columns=['Prediction']),pd.DataFrame(active_probablity,columns=['Active_Prabablity'])],axis=1)
print('XGB_pred\n',preds)
print(r'output saved to = ',output+'xgb_predictions.csv')
# preds.to_csv(output+'xgb_predictions.csv')


'''
# input=data #'C:/Project/Project_3/test_data.csv'
# output= data+'chemprop_predictions.csv'
print('predicting for chemprop model....')
Arguments=['--test_path',data,'--checkpoint_dir',
           'C:/Project/Project_3/model_checkpoint_1_depth_5','--preds_path',output,
           '--features_generator','rdkit_2d_normalized','--no_features_scaling','--ensemble_variance',
           '--no_cuda','--num_workers','0']
args=chemprop.args.PredictArgs().parse_args(Arguments)
chemprop_preds=chemprop.train.make_predictions(args=args)


print('Chemprop_pred\n',chemprop_preds)
# preds.to_csv(output+'xgb_predictions.csv')
'''
print('Predictions complete......')
