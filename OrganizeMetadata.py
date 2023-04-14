import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''Creating a healthy cohort'''
metadata = pd.read_csv('data/Jack_PatientMetaData_052522 - Organized.csv').set_index('PatientID')
healthy = metadata[metadata['Total Disease Count'] == 0].index
print(len(healthy))

betas = pd.read_csv('C:/Users/jack/PycharmProjects/TruDiagnostic/Covariate Models/Blood Type/Data/Selected Data/AllSelectedMethylationData+Age+Logistic.csv')['Unnamed: 0'].values
print(betas)
print(len(betas))

# healthy = list(set(betas) & set(healthy))

popdata = pd.read_csv('data/PopulationData.csv').set_index('Patient ID')
healthy = list(set(healthy) & set(popdata.index))
popdata = popdata.loc[healthy]

print(len(np.unique(healthy)))

print(len(popdata))

# betas = pd.read_csv('C:/Users/jack/PycharmProjects/TruDiagnostic/Covariate Models/Blood Type/data/BetaMatrix_Funnorm_RCP_normalized_betas_1642081709.csv').set_index('Unnamed: 0')

# healthy = metadata[metadata['Total Disease Count']==0].index
# betas.loc[healthy].to_csv('TruHealthyCohort.csv')
# print(len(healthy))

plt.hist(popdata['DunedinPoAm'])
plt.show()
plt.close()
plt.boxplot(popdata['DunedinPoAm'])
plt.show()
plt.close()

average_range = popdata[popdata['DunedinPoAm'] < 1.02][popdata['DunedinPoAm'] > .78]
healthy_range = popdata[popdata['DunedinPoAm'] < .78]
b_average_range = popdata[popdata['DunedinPoAm'] > 1.02]
print(len(average_range))
print(len(healthy_range))
print(len(b_average_range))


plt.hist(average_range['DunedinPoAm'])
plt.show()
plt.close()
plt.boxplot(average_range['DunedinPoAm'])
plt.show()
plt.close()

plt.hist(healthy_range['DunedinPoAm'])
plt.show()
plt.close()
plt.boxplot(healthy_range['DunedinPoAm'])
plt.show()
plt.close()

plt.hist(b_average_range['DunedinPoAm'])
plt.show()
plt.close()
plt.boxplot(b_average_range['DunedinPoAm'])
plt.show()
plt.close()

average_range.to_csv('HealthyAverage.csv')
healthy_range.to_csv('HealthyAboveAverage.csv')
b_average_range.to_csv('BelowAverageRange.csv')
