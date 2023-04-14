import tensorflow as tf
from tensorflow import keras
from keras import layers
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
from glob import glob
import sklearn

tScore_data = pd.read_csv('T-Score Updated NEW.csv').set_index('Patient ID')
print('Total number of t-score data entries: ', len(tScore_data.index))
# id_data = pd.read_csv('PopulationData_081622.csv').set_index('Kit ID')
# valid_indices = list(tScore_data.index.astype(str))
# valid_indices.remove('nan')

# for ID in valid_indices:
#     tScore_data.loc[ID, 'Patient ID'] = id_data.loc[ID, 'Patient ID']
# tScore_data.to_csv('T-Score Updated New.csv')
#
# tScore_data.set_index('Patient ID', inplace=True)
# betas = pd.read_csv('C:/Users/jack/PycharmProjects/TruDiagnostic/Covariate Models/Blood Type/Data/BetaMatrix_Funnorm_RCP_normalized_betas_1642081709.csv').set_index('Unnamed: 0')
# filtered = betas.loc[tScore_data.index]
# filtered.to_csv('VO2MaxBetas.csv')

# print('Original number of t-score data rows: ', len(tScore_data.index))
# shared = list(set(tScore_data.index) & set(pd.read_csv('C:/Users/jack/PycharmProjects/TruDiagnostic/Chronological Age/Data/Chunked Betas/Chunk 1.csv')['Unnamed: 0']))
# betas = pd.DataFrame(index=shared)
# print('New number of t-score data rows: ', len(betas.index))
# files = glob('C:/Users/jack/PycharmProjects/TruDiagnostic/Chronological Age/Data/Chunked Betas/*.csv')
# bad_chunks = []
#
# for file in files:
#     try:
#         print(file)
#         data = pd.read_csv(file).set_index('Unnamed: 0').loc[shared]
#         betas = pd.concat([betas, data], axis=1)
#         print(betas)
#         print(len(betas.columns))
#     except Exception as e:
#         print(e), bad_chunks.append(file)
#
# # betas = pd.read_csv('VO2MaxBetas.csv').set_index('Unnamed: 0')
# print(bad_chunks)
# print(betas, tScore_data)
# betas.to_csv('T-ScoreBetas.csv')
betas = pd.read_csv('T-ScoreBetas.csv').set_index('Unnamed: 0')
print('Numer of original samples: ', len(betas.index))
betas.dropna(inplace=True)
# shared = (set(betas.index) & set(tScore_data.index))
# betas = betas.loc[shared]
# betas.to_csv('T-ScoreBetas.csv')
#
# badPatients = []
# for patient in betas.index:
#     try:
#         t_score = tScore_data.loc[patient, 'T-Score']
#         if np.isnan(t_score):
#             badPatients.append(patient)
#         betas.loc[patient, 'T-Score'] = t_score
#     except Exception as e:
#         print(e, 'Error here')
#
# betas.to_csv('T-ScoreBetas.csv')
# betas.drop(index=badPatients, inplace=True)
# betas.dropna(inplace=True)
# print(betas.columns)

reg = sklearn.linear_model.ElasticNet(alpha=.4, max_iter=10000)
x, y = betas.drop(columns=['T-Score']), betas['T-Score']
print('Number of samples: ', len(x.index))
train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=.1)

print('Elastic net performance: ', cross_val_score(reg, x, y, scoring='neg_mean_absolute_error', cv=4))
reg.fit(train_x, train_y)
preds = reg.predict(val_x)

mae = sklearn.metrics.mean_absolute_error(val_y, preds)
mse = sklearn.metrics.mean_squared_error(val_y, preds)
rsq = sklearn.metrics.r2_score(val_y, preds)
net_error = 0
for p in range(len(preds)):
    pred, real = preds[p], val_y[p]
    error = pred - real
    net_error += error
print('MAE: ', mae)
print('MSE: ', mse)
print('R-Squared: ', rsq)
print('Net error: ', net_error / len(preds))

a, b = np.polyfit(val_y, preds, 1)

print('A, B: ', a, ', ', b)
print('Real: ', val_y)
print('Predicted: ', preds)
plt.plot(val_y, a*val_y+b)
plt.scatter(val_y, preds)
plt.title('T-Score Model Performance')
plt.xlabel('Real Bone Density')
plt.ylabel('Predicted Bone Density')
plt.show()
