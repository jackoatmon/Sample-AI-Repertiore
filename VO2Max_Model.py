import tensorflow as tf
from tensorflow import keras
from keras import layers
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
from glob import glob
import sklearn
import joblib
import category_encoders as ce
import random
from hyperopt import STATUS_OK, hp, tpe, fmin, Trials
import warnings
import os
warnings.filterwarnings('ignore')

'''Customization Parameters'''
new_feature_select = True
new_elastic = True
load_model = False
auto_tune = True

vo2_data = pd.read_csv('VO2Data - UCLA - VO2 Data.csv').set_index('File ID').dropna()
# betas = pd.read_csv('C:/Users/jack/PycharmProjects/TruDiagnostic/Covariate Models/Blood Type/Data/BetaMatrix_Funnorm_RCP_normalized_betas_1642081709.csv').set_index('Unnamed: 0')
# filtered = betas.loc[vo2_data.index]
# filtered.to_csv('VO2MaxBetas.csv')

# shared = list(set(vo2_data.index) & set(pd.read_csv('C:/Users/jack/PycharmProjects/TruDiagnostic/Chronological Age/Data/Chunked Betas/Chunk 1.csv')['Unnamed: 0']))
# betas = pd.DataFrame(index=shared)
# files = glob('C:/Users/jack/PycharmProjects/TruDiagnostic/Chronological Age/Data/Chunked Betas/*.csv')
# bad_chunks = []

# for file in files:
#     try:
#         print(file)
#         data = pd.read_csv(file).set_index('Unnamed: 0').loc[shared]
#         betas = pd.concat([betas, data], axis=1)
#         print(len(betas.columns))
#     except Exception as e:
#         print(e), bad_chunks.append(file)

# betas = pd.read_csv('VO2MaxBetas.csv').set_index('Unnamed: 0')
# print(bad_chunks)
# betas['VO2 Max'] = np.zeros(len(betas.index))
# print(betas, vo2_data)
# betas.to_csv('VO2MaxBetas.csv')


betas = pd.read_csv('VO2MaxBetas.csv').set_index('Unnamed: 0')
betas = betas.transpose()
betas.to_csv('VO2MaxBetas.csv')
# betas = betas.dropna(axis=1)
# # shared = (set(betas.index) & set(vo2_data.index))
# # betas = betas.loc[shared]
# # betas.to_csv('VO2MaxBetas.csv')
# # print(vo2_data, betas, shared)
# print(vo2_data)
# for patient in betas.index:
#     print(patient, ' -- ', vo2_data.loc[patient, 'VO2 Max '])
#     try:
#         betas.loc[patient, 'VO2 Max'] = vo2_data.loc[patient, 'VO2 Max ']
#     except Exception as e:
#         print(e)
# betas.dropna(inplace=True)
# betas.to_csv('VO2MaxBetas.csv')

print(betas.columns)

if new_feature_select:
    importances = mutual_info_regression(betas.iloc[:, 1:], betas.loc[:, 'VO2 Max'].astype(float))
    feat_importances = pd.Series(importances, betas.columns[1:])

    average_importance = np.average(feat_importances.values)
    print('Average relevance: ', average_importance)

    threshold = 2 * average_importance
    relevant_cpgs = feat_importances[feat_importances['0'] > threshold]
    # if mutual:
    #     relevant_cpgs.to_csv('data/selected data/RelevantCpGs.csv')
    # else:
    #     relevant_cpgs.to_csv('data/selected data/p-valRelevantCpGs.csv')

    cpgs_target = list(relevant_cpgs.index)
    cpgs_target.append('Chronological Age')
    if not new_elastic:
        cpgs_target.append('Elastic Predictions')
    betas = betas[cpgs_target]
    betas.sort_index(axis=1, inplace=True)
    betas.to_csv('data/selected data/SelectedMethylationData NEW.csv')
    print('Final Number of CpGs: ', len(cpgs_target))


y = betas['VO2 Max']
x = betas.drop(columns=['VO2 Max'])
test_size = .1
train_x_labeled, test_x_labeled, train_y_labeled, test_y_labeled = train_test_split(x, y, test_size=test_size)
train_x_labeled, val_x_labeled, train_y_labeled, val_y_labeled = train_test_split(train_x_labeled, train_y_labeled, test_size=test_size)

if new_elastic:
    print('Beginning elastic net training...')
    reg = sklearn.linear_model.ElasticNet(alpha=.3, max_iter=10000)
    reg.fit(train_x_labeled, train_y_labeled)
else:
    reg = joblib.load('VO2Max_ElasticModel.joblib')
    print('Elastic net performance: ', cross_val_score(reg, x, y, scoring='neg_mean_absolute_error', cv=4))


preds = reg.predict(val_x_labeled)

mae = sklearn.metrics.mean_absolute_error(val_y_labeled, preds)
mse = sklearn.metrics.mean_squared_error(val_y_labeled, preds)
rsq = sklearn.metrics.r2_score(val_y_labeled, preds)


net_error = 0
for p in range(len(preds)):
    pred, real = preds[p], float(val_y_labeled[p])
    error = pred - real
    net_error += error
print('MAE: ', mae)
print('MSE: ', mse)
print('R-Squared: ', rsq)
print('Net error: ', net_error / len(preds))

a, b = np.polyfit(val_y_labeled, preds, 1)

print('A, B: ', a, ', ', b)
print('Real: ', val_y_labeled)
print('Predicted: ', preds)
plt.plot(val_y_labeled, a*val_y_labeled+b)
plt.scatter(val_y_labeled, preds)
plt.title('VO2 Max Model Performance')
plt.xlabel('Real VO2 Max')
plt.ylabel('Predicted VO2 Max')
plt.show()

scaler = StandardScaler()
train_x, train_y = scaler.fit_transform(train_x_labeled), np.array(train_y_labeled)
test_x, test_y = scaler.transform(test_x_labeled), np.array(test_y_labeled)
val_x, val_y = scaler.transform(val_x_labeled), np.array(val_y_labeled)

patient_order = betas.index

print('Overall patient ordrer: ', patient_order)

'''Metrics and Hyperparameters'''
metrics = [keras.metrics.MeanAbsoluteError(name='MAE'),
           keras.metrics.MeanSquaredError(name='MSE')]

batch_size = 256
epochs = 250
# learning_rate = .01
# num_layers = 5
# layer_size = 50
# noise = .05
# l1, l2 = .01, .1

# Current best: ChronoAge0.01-0.0019247923925306844-8.0-93.0-5.0-0.05635867926717027-0.034533334567416266-0.08505781598392871-0.011274876020738067
learning_rates = [.01]  # , .001]
min_lrs = [.001, .0005]
patiences = [14]
layer_sizes = [125]  # , 125]
num_layers2 = [7]  # 7 current best
noises = [.4]  # .4 current best
l1s = [.005]  # .005 current best
l2s = [0.001]  # .01 current best
b1s = [.99]  # [.5, .9, .99]
b2s = [.999]  # [.5, .9, .99]
dropout_rates = [.3]   # .3 current best
# df_log = pd.read_csv('data/performances/GridSearchPerformance.csv').set_index('Unnamed: 0')
df_log = pd.DataFrame(
    columns=['Learning Rate', 'Min Learning Rate', 'Patience', 'Layer Size', 'Number of Layers', 'Noise', 'L1', 'L2',
             'Dropout Rate', 'Val_RMSE', 'Val_MAE'])

if load_model:
    model_name = 'ChronoAge0.005-125-9-0.01-0.07-0.1'
    model = keras.models.load_model('models/' + model_name)
else:
    if auto_tune:
        space = {'min_lr': hp.uniform('min_lr', .0001, .05),
                 'patience': hp.quniform('patience', 8, 15, 1),
                 'layer_size': hp.quniform('layer_size', 80, 150, 1),
                 'num_layers': hp.quniform('num_layers', 4, 8, 1),
                 'noise': hp.uniform('noise', .01, .4),
                 'l1': hp.uniform('l1', 0, .09),
                 'l2': hp.uniform('l2', 0, .09),
                 'dropout_rate': hp.uniform('dropout_rate', .01, .4)}
        hyper_algorithm = tpe.suggest

        def tuning_objective(hyperparameters={}):
            min_lr, patience, layer_size, num_layers, noise, l1, l2, dropout_rate = hyperparameters['min_lr'], \
                                                                                    hyperparameters['patience'],\
                                                                                    hyperparameters['layer_size'],\
                                                                                    hyperparameters['num_layers'],\
                                                                                    hyperparameters['noise'], \
                                                                                    hyperparameters['l1'], \
                                                                                    hyperparameters['l2'], \
                                                                                    hyperparameters['dropout_rate']
            model_name = 'ChronoAge' + str(.01) + '-' + str(min_lr) \
                         + '-' + str(patience) + '-' + str(layer_size) + '-' \
                         + str(num_layers) + '-' + str(noise) + '-' + str(l1) \
                         + '-' + str(l2) + '-' + str(dropout_rate)

            file_path = 'models/' + model_name
            print(model_name)

            callbacks = [keras.callbacks.EarlyStopping(monitor='loss',
                                                       patience=patience,
                                                       mode='min',
                                                       restore_best_weights=True),
                         keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                           patience=int(patience / 2),
                                                           factor=.5,
                                                           verbose=1,
                                                           mode='min',
                                                           min_lr=min_lr), ]

            opt = keras.optimizers.Adam(learning_rate=.01, beta_1=.99,
                                        beta_2=.999)

            mod = keras.Sequential(layers.Dense(len(train_x[0]), activation='relu'))

            for layer in range(int(num_layers)):
                mod.add(keras.layers.BatchNormalization())
                mod.add(keras.layers.GaussianNoise(noise))
                mod.add(keras.layers.ActivityRegularization(l1=l1, l2=l2))
                mod.add(keras.layers.Dense(layer_size, activation='relu'))
                mod.add(keras.layers.Dropout(dropout_rate))

            output_layer = mod.add(keras.layers.Dense(1))  # , bias_initializer=initial_bias))

            mod.compile(optimizer=opt,
                          loss='mean_squared_error',
                          metrics=metrics)

            mod.fit(x=train_x,
                      y=train_y,
                      validation_data=[test_x, test_y],
                      batch_size=batch_size,
                      epochs=epochs,
                      callbacks=callbacks,
                      verbose=1)

            mod.save(file_path)
            print('Save path of model: ', file_path)

            loss = mod.evaluate(val_x, val_y)

            print('Parameters: ', min_lr, patience, layer_size, num_layers, noise, l1, l2, dropout_rate)
            print('Loss: ', loss, '\n')

            return {'Loss': loss, 'Params': hyperparameters, 'Status': STATUS_OK}

        best = fmin(fn=tuning_objective, space=space, algo=tpe.suggest,
                    max_evals=200, trials=Trials())

    else:
        for learning_rate in learning_rates:
            for min_lr in min_lrs:
                for patience in patiences:
                    for layer_size in layer_sizes:
                        for num_layers in num_layers2:
                            for noise in noises:
                                for l1 in l1s:
                                    for l2 in l2s:
                                        for b1 in b1s:
                                            for b2 in b2s:
                                                for dropout_rate in dropout_rates:
                                                    model_name = 'ChronoAge' + str(learning_rate) + '-' + str(min_lr)\
                                                                 + '-' + str(patience) + '-' + str(layer_size) + '-' \
                                                                 + str(num_layers) + '-' + str(noise) + '-' + str(l1)\
                                                                 + '-' + str(l2) + '-' + str(dropout_rate)

                                                    file_path = 'models/' + model_name
                                                    print(model_name)

                                                    callbacks = [keras.callbacks.EarlyStopping(monitor='loss',
                                                                                               patience=patience,
                                                                                               mode='min',
                                                                                               restore_best_weights=True),
                                                                 keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                                                                   patience=int(patience/2),
                                                                                                   factor=.5,
                                                                                                   verbose=1,
                                                                                                   mode='min',
                                                                                                   min_lr=min_lr),]
                                                                 # keras.callbacks.ModelCheckpoint(
                                                                 #     filepath=file_path,
                                                                 #     monitor='MAE',
                                                                 #     mode='min',
                                                                 #     save_freq=100)]

                                                    opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=b1,
                                                                                beta_2=b2)

                                                    model = keras.Sequential(layers.Dense(len(train_x[0]), activation='relu'))

                                                    for layer in range(num_layers):
                                                        model.add(keras.layers.BatchNormalization())
                                                        model.add(keras.layers.GaussianNoise(noise))
                                                        model.add(keras.layers.ActivityRegularization(l1=l1, l2=l2))
                                                        model.add(keras.layers.Dense(layer_size, activation='relu'))
                                                        model.add(keras.layers.Dropout(dropout_rate))

                                                    output_layer = model.add(keras.layers.Dense(1))  # , bias_initializer=initial_bias))

                                                    model.compile(optimizer=opt,
                                                                  loss='mean_squared_error',
                                                                  metrics=metrics)

                                                    model.fit(x=train_x,
                                                              y=train_y,
                                                              validation_data=[test_x, test_y],
                                                              batch_size=batch_size,
                                                              epochs=epochs,
                                                              callbacks=callbacks,
                                                              verbose=1)

                                                    model.save(file_path)

                                                    print('Save path of model: ', file_path)

                                                    mets = model.evaluate(test_x, test_y)

                                                    df_log.loc[model_name, ['Learning Rate', 'Min Learning Rate',
                                                                            'Patience', 'Layer Size', 'Number of Layers',
                                                                            'Noise', 'L1', 'L2', 'Dropout Rate',
                                                                            'Val_RMSE', 'Val_MAE']] = \
                                                                            learning_rate, min_lr, patience, layer_size, num_layers, \
                                                                            noise, l1, l2, dropout_rate, np.sqrt(mets[0]), mets[1]

                                                    print(df_log.loc[model_name])

                                                    df_log.to_csv('data/performances/GridSearch' + model_name + '.csv')

                                                    validation_output = model.predict(test_x)

                                                    predictionary = {}
                                                    for i in range(len(validation_output)):
                                                        patient = test_y_labeled.index[i]
                                                        predicted_age = validation_output[i]
                                                        real_age = test_y_labeled[patient]
                                                        error = real_age - predicted_age

                                                        predictionary[patient] = [predicted_age, real_age, error]

                                                    print('\n\n*************************')
                                                    print('Validation performance...')
                                                    val_accuracy = model.evaluate(val_x, val_y)
                                                    val_output = model.predict(val_x).flatten()

                                                    pearson = stats.pearsonr(val_output, val_y)
                                                    spearman = stats.spearmanr(val_output, val_y)
                                                    rsquared = sklearn.metrics.r2_score(val_y, val_output)

                                                    print('valnib order: ', val_y_labeled.index)

                                                    # print('valnib elastic r-squared: ', reg.score(val_elastic_x, val_y))

                                                    print('Model evaluation (MSE, MAE): ', val_accuracy[:2])
                                                    print('R-squared : ', rsquared)
                                                    print('MAE confirmation: ', sklearn.metrics.mean_absolute_error(val_y, val_output))
                                                    print('RMSE confirmation: ', np.sqrt(sklearn.metrics.mean_squared_error(val_y, val_output)))
                                                    print('Pearson correlation: ', pearson)
                                                    print('Spearman correlation: ', spearman)
                                                    print('R-value confirmation: ', np.corrcoef(val_output, val_y)[0][1])
                                                    print('\n\n*************************')
            else:
                learning_rate = .01  # , .001]
                min_lr = .001
                patience = 8
                layer_size = 100  # , 125]
                num_layers = 7
                noise = .15
                l1 = .06  # [.2, .1]
                l2 = 0  # [.2, .1]
                b1 = .99  # [.5, .9, .99]
                b2 = .999  # [.5, .9, .99]
                dropout_rate = .08

                model_name = 'ChronoAge' + str(learning_rate) + '-' + str(min_lr) \
                             + '-' + str(patience) + '-' + str(layer_size) + '-' \
                             + str(num_layers) + '-' + str(noise) + '-' + str(l1) \
                             + '-' + str(l2) + '-' + str(dropout_rate)

                print(model_name)

                callbacks = [keras.callbacks.EarlyStopping(monitor='loss',
                                                           patience=patience,
                                                           mode='min',
                                                           restore_best_weights=True),
                             keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                               patience=int(patience / 2),
                                                               factor=.5,
                                                               verbose=1,
                                                               mode='min',
                                                               min_lr=min_lr),
                             keras.callbacks.ModelCheckpoint(
                                 filepath='models/' + model_name,
                                 monitor='MAE',
                                 mode='min',
                                 save_freq=100)]

                opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=b1,
                                            beta_2=b2)

                model = keras.Sequential(layers.Dense(len(train_x[0]), activation='relu'))

                for layer in range(num_layers):
                    model.add(keras.layers.BatchNormalization())
                    model.add(keras.layers.GaussianNoise(noise))
                    model.add(keras.layers.ActivityRegularization(l1=l1, l2=l2))
                    model.add(keras.layers.Dense(layer_size, activation='relu'))
                    model.add(keras.layers.Dropout(dropout_rate))

                output_layer = model.add(keras.layers.Dense(1))  # , bias_initializer=initial_bias))

                model.compile(optimizer=opt,
                              loss='mean_squared_error',
                              metrics=metrics)

                model.fit(x=train_x,
                          y=train_y,
                          validation_data=[test_x, test_y],
                          batch_size=batch_size,
                          epochs=epochs,
                          callbacks=callbacks,
                          verbose=1)

                mets = model.evaluate(test_x, test_y)

                model.save('models/' + model_name)

                df_log.loc[model_name, ['Learning Rate', 'Min Learning Rate',
                                        'Patience', 'Layer Size', 'Number of Layers',
                                        'Noise', 'L1', 'L2', 'Dropout Rate',
                                        'Val_RMSE', 'Val_MAE']] = \
                    learning_rate, min_lr, patience, layer_size, num_layers, \
                    noise, l1, l2, dropout_rate, np.sqrt(mets[0]), mets[1]

                print(df_log.loc[model_name])

                df_log.to_csv('data/performances/GridSearch' + model_name + '.csv')
