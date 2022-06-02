from Data_preprocessing import *
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import tensorflow as tf

remote_sensing_data = 'D:/programes/dataset/aster-finalstack2.tif'
trainingDS = 'D:/programes/qgis/train_reg.shp'
testingDS = 'D:/programes/qgis/test_reg.shp'

band_data = rs_preprocessing(remote_sensing_data, reshape=False)
x_train, y_train = dataFitting(remote_sensing_data, band_data, trainingDS)
x_test, y_test = dataFitting(remote_sensing_data,band_data, testingDS)
# print(x_train)
# print(x_test)

def norm(Data):
    norm = np.linalg.norm(Data)
    normData = Data/norm
    return normData

x_train_norm = norm(x_train)
x_test_norm = norm(x_test)
print(x_train_norm)
print(x_test_norm)

ann_training = tfPipline(x_train_norm, y_train, shuffle=False, repeat=True)
ann_testing = tfPipline(x_test_norm, '', shuffle=False, repeat=False, BUFFER_SIZE=0, BATCH_SIZE=32)

reset_random_seeds()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(42,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.summary()

history2 = model.fit(ann_training, epochs=100, steps_per_epoch=x_train.shape[0]/64)
ann_predictions = model.predict(ann_testing).flatten()

print(ann_predictions)
print(y_test)

annauc = roc_auc_score(y_test, ann_predictions)
print(annauc)
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
