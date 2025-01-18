In this project, binary classifier machine learning models were developed to predict whether 
a given molecule has the potential of being an anti-leishmanial or not against Leishmania 
donovani. The predictions are based on the use of features such as molecular descriptors 
values and molecular fingerprints such as ECFP and FCFP calculated using RDKit software 
from SMILES strings of compounds. The machine learning models were trained on 
collected data of 5705 drug-like compounds. After analyzing the models developed it was 
found that 2 models amongst the five, namely Extreme gradient boost (XGB), and Random 
Forest (RF), outperform the others in terms of accuracy. After that, the Gini importance 
feature selection method was used to pick the best-performing descriptors. These models 
performed with more than 79% accuracy in both the cross-validation and test sets even with 
a smaller subset of descriptors. A web tool was also developed with these models. 

I hereby declare that the project entitled, “From Chemical Structures to Anti
Leishmanial Activity: A Machine Learning Approach”, submitted in partial 
fulfillment of the requirements of the degree of Master of Science in 
Bioinformatics, has been carried out by me at Bioinformatics Centre, Savitribai 
Phule Pune University
