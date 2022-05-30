from sklearn import preprocessing
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from Data_preprocessing import *
import matplotlib.pyplot as plt

remote_sensing_data = 'D:/programes/dataset/aster-finalstack2.tif'
trainingDS = 'D:/programes/qgis/train_reg.shp'
testingDS = 'D:/programes/qgis/test_reg.shp'

band_data = rs_preprocessing(remote_sensing_data, reshape=False)
x_train, y_train = dataFitting(remote_sensing_data, band_data, trainingDS)
x_test, y_test = dataFitting(remote_sensing_data,band_data, testingDS)
# print(x_train)
# print(x_test)
x_train_norm = preprocessing.normalize(x_train)
x_test_norm = preprocessing.normalize(x_test)
# print(x_train_norm)
# print(x_test_norm)

# reset_random_seeds()
rfmodel = RandomForestRegressor(n_estimators=500, oob_score=True, verbose=1, n_jobs=-1, min_samples_split=4)

rfmodel.fit(x_train_norm, y_train)
y_predicted = rfmodel.predict(x_test_norm)

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

