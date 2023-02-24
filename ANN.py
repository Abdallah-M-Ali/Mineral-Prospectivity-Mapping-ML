from Data_preprocessing import *
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, cohen_kappa_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import RMSprop, Adam
from keras.models import Sequential
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV
# from keras.wrappers.scikit_learn import KerasRegressor
from scikeras.wrappers import KerasRegressor




# remote_sensing_data = 'D:/Graduation/data/Sentinel-2/Sentinel2_fullEL.tiff'
# remote_sensing_data = 'D:/Graduation/data/ASTER/ASTER_fullEL2.tiff'
# remote_sensing_data = 'D:/Graduation/data/Landsat-8/Landsat8_fullEL.tiff'
remote_sensing_data = 'D:/Graduation/data/Integration/RF_Integration.tiff'
# remote_sensing_data = 'D:/Graduation/data/Integration/Full_Integration.tiff'
sampleData = 'D:/Graduation/data/samples_point_exp.shp'
trainDirectory = 'D:/Graduation/data/Geological/training.shp'
testDirectory = 'D:/Graduation/data/Geological/testing.shp'

##use the Data-preprocessing file to manipulate the data as ML input
band_data, img_as_array = rs_preprocessing(remote_sensing_data, reshape=True)
x_train, y_train = dataFitting(remote_sensing_data, band_data, trainDirectory)
x_test, y_test = dataFitting(remote_sensing_data,band_data, testDirectory)
# print(x_train)
# print(x_test)

# def norm(Data):
#     norm = np.linalg.norm(Data)
#     normData = Data/norm
#     return normData

# x_train_norm = norm(x_train)
# x_test_norm = norm(x_test)
# print(x_train_norm)
# print(x_test_norm)

# ann_training = tfPipline(x_train, y_train, shuffle=False, repeat=True)
# ann_testing = tfPipline(x_test, '', shuffle=False, repeat=False, BUFFER_SIZE=0, BATCH_SIZE=32)
# ann_training = tfPipline(x, '', shuffle=False, repeat=False)
# ann_testing = tfPipline(y, '', shuffle=False, repeat=False)


reset_random_seeds()

#Defining the model structure as function to be used for Grid Search
def define_model (neurons_num=64, activation='relu', learning_rate=0.01):
    model = Sequential()
    model.add(Flatten(input_shape=(16,)))
    model.add(Dense(neurons_num, activation=activation))
    model.add(Dense(neurons_num, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    optimizing = RMSprop(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizing, metrics=['accuracy'])
    return model

#parameters selection
epochs = 200
# batch_size = [2, 5, 10, 14, 18, 20, 30, 35]
batch_size = [35]
learning_rate = [0.01, 0.001]
# learning_rate = [0.0001, 0.001, 0.01, 0.1]
# activation = ['relu', 'sigmoid']
neurons_num = [4, 8, 16, 32]
# neurons_num = [32]

model1 = KerasRegressor(model=define_model, epochs=epochs, batch_size=batch_size, neurons_num=neurons_num,
                        learning_rate=learning_rate, verbose=1)


param_grid = dict(neurons_num=neurons_num, learning_rate=learning_rate, batch_size=batch_size)
grid = GridSearchCV(estimator=model1, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=1, cv=5)
# scoring='roc_auc',
model = grid.fit(x_train, y_train)
print('best result', model.best_score_, 'from', model.best_params_)
df = pd.DataFrame(model.cv_results_)
# df.to_excel('D:/Graduation/data/Integration/output_FullIN/sta/ANN_sta_Neurons.xlsx')
columns = df.columns.values.tolist()
print(columns)
df_sorted = df[['param_learning_rate', 'param_neurons_num', 'param_batch_size', 'mean_test_score']]
print(df_sorted)



#model prediction after training
ann_predictions = model.predict(x_test).flatten()
round_prediction = [round(i) for i in ann_predictions]

print(ann_predictions)
print(round_prediction)
print(y_test)
# print(acc)
annauc = roc_auc_score(y_test, ann_predictions)
print(annauc)


#ploting the results including the ROC curve and the Confusion  matrix
fpr, tpr, _ = roc_curve(y_test, ann_predictions)
plt.plot(fpr,tpr, color ='red', lw=2, label="All Variables")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-.05, 1.0])
plt.ylim([-.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

table = pd.DataFrame(confusion_matrix(y_test, round_prediction))
print(classification_report(y_test, round_prediction))
print('K = ', cohen_kappa_score(y_test, round_prediction))

plot = sns.heatmap(table,annot=True, fmt='d', edgecolor="blue")
# plot = sns.heatmap(table,annot=True, cmap='viridis')
plt.show()

##writing raster of the prediction
try:
    class_prediction = model.predict(img_as_array).flatten()
except MemoryError:
    slices = int(round(len(img_as_array) / 2))
    test = True
    while test == True:
        try:
            class_preds = list()

            temp = model.predict(img_as_array[0:slices + 1, :])
            class_preds.append(temp)

            for i in range(slices, len(img_as_array), slices):
                print('{} %, derzeit: {}'.format((i * 100) / (len(img_as_array)), i))
                temp = model.predict(img_as_array[i + 1:i + (slices + 1), :])
                class_preds.append(temp)

        except MemoryError as error:
            slices = slices / 2
            print('Not enought RAM, new slices = {}'.format(slices))

        else:
            test = False
else:
    print('Class prediction was successful without slicing!')

class_prediction = class_prediction.reshape(band_data[:, :, 0].shape)
print('Reshaped back to {}'.format(class_prediction.shape))

mask = np.copy(band_data[:,:,0])
mask[mask > 0.0] = 1.0 # all actual pixels have a value of 1.0

class_prediction.astype(np.float16)
class_prediction_ = class_prediction*mask

plt.subplot(121)
plt.imshow(class_prediction, cmap=plt.cm.Spectral)
plt.title('classification unmasked')

plt.subplot(122)
plt.imshow(class_prediction_, cmap=plt.cm.Spectral)
plt.title('classification masked')

plt.show()

##save the output as tiff if you like the results
# output_image = 'D:/Graduation/data/Integration/output_RFIN/ANN_RFInt.tiff'
# write_raster(remote_sensing_data, class_prediction, band_data, output_image)


del model1
