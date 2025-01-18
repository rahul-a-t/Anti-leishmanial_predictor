import pickle
from flask import Flask, render_template, request,Response
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from rdkit.Chem import Descriptors,AllChem
from rdkit import Chem
import pandas as pd
import numpy as np


def converting_SMILES_to_input(smiles=list):
  # converts list of SMILES into a dataframe for input in the model ecfp fingerprints+ rdkit 2d descriptors which are (Z-scale normalixzed)
  ecfp_l=[]
  descriptors_list=[]
  desc_names=[]
  for i in Descriptors.descList:
    # print(i)
    desc_names.append(i[0])
  for i in smiles:
    # print(i)
    mol = Chem.MolFromSmiles(i) # convert SMILES to RDKit molecule
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
    ecfp_l.append(ecfp)
    desc_list=[]
    for desc, func in Descriptors.descList:
        # print('desc',desc, func)
        desc_list.append(func(mol))
    descriptors_list.append(desc_list)
  descriptors_l_normalized = scaler.transform(descriptors_list)
  input_dataframe= pd.concat([pd.DataFrame(np.array(ecfp_l)),pd.DataFrame(descriptors_l_normalized,columns=desc_names)],axis=1)
  input_dataframe.columns = input_dataframe.columns.astype(str)
  return input_dataframe




app = Flask(__name__)
xgb_ecfp_2d_desc = pickle.load(open('ML_workflow\models\main_model_xgb_ecfp_2d_desc.pkl', 'rb'))
scaler = pickle.load(open('ML_workflow\models\desc_scaler.pkl', 'rb'))

pipeline = Pipeline([
    ('transformer', FunctionTransformer(converting_SMILES_to_input)),
    ('classifier', xgb_ecfp_2d_desc)
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    smiles = request.form['smiles'].split('\n')
    xgb_prediction = pipeline.predict(smiles)
    xgb_prediction_proba = pipeline.predict_proba(smiles)
    return render_template('result.html',smiles=smiles, prediction=xgb_prediction,prediction_proba=xgb_prediction_proba)

def csv_generator(data):
    yield ','.join(['SMILE','Active Probablity','Predicted Activity']) + '\n'
    for smile,prediction_proba, prediction in data:
        yield ','.join([smile,prediction_proba[1], 'Active' if prediction == 1 else 'Inactive']) + '\n'



@app.route('/download_csv')
def download_csv():
    smiles = request.args.getlist('smiles')
    prediction_proba=request.args.getlist('prediction_proba')
    prediction = request.args.getlist('prediction')
    data = zip(smiles,prediction_proba, prediction)
    response = Response(
        csv_generator(data),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=result.csv'}
    )
    return response


if __name__=="__main__":
    app.run(debug=True)

