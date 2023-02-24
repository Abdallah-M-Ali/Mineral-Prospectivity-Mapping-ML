from sklearn import preprocessing
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, cohen_kappa_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from Data_preprocessing import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import make_scorer, mean_absolute_error
import os
import random


# remote_sensing_data = 'D:/Graduation/data/Sentinel-2/Sentinel2_fullEL.tiff'
# remote_sensing_data = 'D:/Graduation/data/Landsat-8/Landsat8_fullEL.tiff'
remote_sensing_data = 'D:/Graduation/data/Integration/RF_Integration.tiff'
# remote_sensing_data = 'D:/Graduation/data/Integration/Full_Integration.tiff'
sampleData = 'D:/Graduation/data/samples_point_exp.shp'
trainDirectory = 'D:/Graduation/data/Geological/training.shp'
testDirectory = 'D:/Graduation/data/Geological/testing.shp'
# trainingDS = 'D:/programes/qgis/train_reg.shp'
# testingDS = 'D:/programes/qgis/test_reg.shp'

# target_variable(sampleData, trainDirectory, testDirectory, 0.7)


band_data, img_as_array = rs_preprocessing(remote_sensing_data, reshape=True)
x_train, y_train = dataFitting(remote_sensing_data, band_data, trainDirectory)
x_test, y_test = dataFitting(remote_sensing_data,band_data, testDirectory)
# print(x_train)
print(y_test)
# x_train_norm = preprocessing.normalize(x_train)
# x_test_norm = preprocessing.normalize(x_test)
# print(x_train_norm)
# print(x_test_norm)


reset_random_seeds()




rf = RandomForestRegressor(n_jobs=-1)
# n_estimators=500, oob_score=True, verbose=1, n_jobs=-1, min_samples_split=4
min_samples_split = [2, 4, 6, 8, 10, 12]
# min_samples_split = [6]
n_estimators = [50, 100, 200, 250, 300, 400, 500]
# n_estimators = [300]
# neg_mean_absolute_error = make_scorer(mean_absolute_error, greater_is_better=False)

grid = GridSearchCV(estimator=rf, param_grid=dict(n_estimators=n_estimators, min_samples_split=min_samples_split),
                    scoring='neg_mean_squared_error', n_jobs=1, cv=5)
rfmodel = grid.fit(x_train, y_train)
print('best result', rfmodel.best_score_, 'from', rfmodel.best_params_)
df = pd.DataFrame(rfmodel.cv_results_)
df_sorted = df[['param_min_samples_split', 'param_n_estimators', 'mean_test_score']]
# df.to_excel('D:/Graduation/data/Sentinel-2/output/sta/RF_sta.xlsx')
# df.to_excel('D:/Graduation/data/Integration/output_RFIN/sta/RF_sta.xlsx')
# df.to_excel('D:/Graduation/data/Integration/output_FullIN/sta/RF_sta.xlsx')
# print(df_sorted)


y_predicted = rfmodel.predict(x_test)



round_prediction = [round(i) for i in y_predicted]
# print(k_fold, "mean is", k_fold.mean(), "std is", k_fold.std())
print(y_predicted)
print(round_prediction)
print(y_test)


auc = roc_auc_score(y_test, y_predicted)
print(auc)
fpr, tpr, _ = roc_curve(y_test, y_predicted)
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

plot = sns.heatmap(table, annot=True, fmt='d', cmap="Blues")
# plot = sns.heatmap(table, annot=True, cmap='viridis')
plt.show()

# feature_importance = pd.DataFrame(rf.feature_importances_, columns=['importance'])
# featureName = ['NE_Fualt', 'NW_Fualt', 'Lineament', 'Intrusion', 'PC4_Argillic', 'PC4_Phyllic', 'PC3_Propylitic', 'PC4_OHbearing', 'PC2_IronOides',
#                'BR_2/1', 'BR_4/5', 'BR_4/6', 'BR_4/7', 'RBD1_Argillic', 'RBD2_Phyllic', 'RBD3', 'RBD4',
#                'ALI', 'CLI', 'KAI', 'OHI', 'MNF1', 'MNF2', 'MNF3', 'MNF4']
# feature_importance['Name'] = featureName
# features_order = feature_importance.sort_values(by='importance', ascending=False)
#
#
# # print(feature_importance)
#
# fig, ax = plt.subplots()
# plt.bar(features_order['Name'], features_order['importance'])
# plt.xticks(rotation='vertical')
# # feature_importance.plot.bar(ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()
# plt.show()

try:
    class_prediction = rfmodel.predict(img_as_array)
except MemoryError:
    slices = int(round(len(img_as_array) / 2))
    test = True
    while test == True:
        try:
            class_preds = list()

            temp = rfmodel.predict(img_as_array[0:slices + 1, :])
            class_preds.append(temp)

            for i in range(slices, len(img_as_array), slices):
                print('{} %, derzeit: {}'.format((i * 100) / (len(img_as_array)), i))
                temp = rfmodel.predict(img_as_array[i + 1:i + (slices + 1), :])
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
#
# output_image = 'D:/Graduation/data/Sentinel-2/output/RF2_Sentinel2.tiff'
# output_image = 'D:/Graduation/data/Integration/output_FullIN/RF_FullInt.tiff'
output_image = 'D:/Graduation/data/Integration/output_RFIN/RF_RFInt.tiff'
write_raster(remote_sensing_data, class_prediction, band_data, output_image)

del rfmodel
