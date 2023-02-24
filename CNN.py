from Data_preprocessing import *
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import RMSprop, Adam
from keras.models import Sequential
# from keras.wrappers.scikit_learn import KerasRegressor
from scikeras.wrappers import KerasRegressor
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, cohen_kappa_score, classification_report


# remote_sensing_data = 'D:/Graduation/data/ASTER/ASTER_fullEL2.tiff'
# remote_sensing_data = 'D:/Graduation/data/Sentinel-2/Sentinel2_fullEL.tiff'
# remote_sensing_data = 'D:/Graduation/data/Landsat-8/Landsat8_fullEL.tiff'
remote_sensing_data = 'D:/Graduation/data/Integration/RF_Integration.tiff'
# remote_sensing_data = 'D:/Graduation/data/Integration/Full_Integration.tiff'
trainingDS = 'D:/Graduation/data/Geological/training.shp'
testingDS = 'D:/Graduation/data/Geological/testing.shp'

band_data, img_as_array = rs_preprocessing(remote_sensing_data, reshape=True)
x_train, y_train = dataFitting(remote_sensing_data, band_data, trainingDS)
x_test, y_test = dataFitting(remote_sensing_data,band_data, testingDS)
# print(x_train)
# print(x_test)

# def norm(Data):
#     norm = np.linalg.norm(Data)
#     normData = Data/norm
#     return normData
#
# x_train_norm = norm(x_train)
# x_test_norm = norm(x_test)
# print(x_train_norm)
# print(x_test_norm)
#
cnn_reshaped = cnn_input(x_train)
cnn_test_reshaped = cnn_input(x_test)

# cnn_reshaped = cnn_reshaped.astype(np.float64)
# cnn_test_reshaped = cnn_test_reshaped.astype(np.float64)

print(cnn_reshaped.dtype, " for training", cnn_test_reshaped.dtype, "for testing")
print(cnn_reshaped.shape, cnn_test_reshaped.shape)

# cnn_training = tfPipline(cnn_reshaped, y_train, shuffle=False, repeat=True)
# cnn_testing = tfPipline(cnn_test_reshaped, '', shuffle=False, repeat=False, BUFFER_SIZE=0, BATCH_SIZE=32)

reset_random_seeds()


def define_model (neurons_num=16, activation='relu', learning_rate=0.01, kernel=16):
    model = Sequential()
    model.add(Conv1D(kernel, 3, activation='relu', kernel_initializer='random_normal', input_shape=(16, 1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(kernel, 3, activation='relu'))
    # model.add(MaxPooling1D(2))
    model.add(Conv1D(kernel, 3, activation='relu'))
    # model.add(MaxPooling1D(2))
    # model.add(Conv1D(kernel, 3, activation='relu'))
    # model.add(MaxPooling1D(2))

    model.add(Flatten())
    model.add(Dense(neurons_num, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    optimizing = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizing, metrics=['accuracy'])
    model.summary()
    return model

epochs = 200
batch_size = [14, 18, 20, 30, 35]
# batch_size = [35]
# learning_rate = [0.0001, 0.001, 0.01, 0.1]
learning_rate = [0.001]
# activation = ['relu', 'sigmoid']
# neurons_num = [4, 8, 16, 32, 64, 128]
neurons_num = [64]
# kernel = [4, 8, 16, 32, 64, 128]
kernel = [64]

model1 = KerasRegressor(build_fn=define_model, epochs=epochs,learning_rate=learning_rate, kernel=kernel,
                        batch_size=batch_size, neurons_num=neurons_num, verbose=1)



param_grid = dict(learning_rate=learning_rate, kernel=kernel, batch_size=batch_size, neurons_num=neurons_num)
grid = GridSearchCV(estimator=model1, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=1, cv=5)

# cnn_history = model.fit(cnn_training, epochs=100, steps_per_epoch=x_train.shape[0]/64)
model = grid.fit(cnn_reshaped, y_train)

print('best result', model.best_score_, 'from', model.best_params_)
df = pd.DataFrame(model.cv_results_)
# df.to_excel('D:/Graduation/data/Integration/output_FullIN/sta/CNN_sta_Batch.xlsx')


cnn_predictions = model.predict(cnn_test_reshaped)
print(cnn_predictions)
print(y_test)
round_prediction = [round(i) for i in cnn_predictions]
print(round_prediction)



cnnauc = roc_auc_score(y_test, cnn_predictions)
print(cnnauc)
fpr, tpr, _ = roc_curve(y_test, cnn_predictions)
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



img_as_array = cnn_input(img_as_array)

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


# output_image = 'D:/Graduation/data/Integration/output_FullIN/CNN_FullInt2.tiff'
output_image = 'D:/Graduation/data/Integration/output_RFIN/CNN_RFInt.tiff'
write_raster(remote_sensing_data, class_prediction, band_data, output_image)

del model1