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

cnn_reshaped = cnn_input(x_train_norm)
cnn_test_reshaped = cnn_input(x_test_norm)

cnn_reshaped = cnn_reshaped.astype(np.float64)
cnn_test_reshaped = cnn_test_reshaped.astype(np.float64)

print(cnn_reshaped.dtype, " for training", cnn_test_reshaped.dtype, "for testing")
print(cnn_reshaped.shape, cnn_test_reshaped.shape)

cnn_training = tfPipline(cnn_reshaped, y_train, shuffle=False, repeat=True)
cnn_testing = tfPipline(cnn_test_reshaped, '', shuffle=False, repeat=False, BUFFER_SIZE=0, BATCH_SIZE=32)

reset_random_seeds()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(42, 1)),
    # tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    # tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    # tf.keras.layers.MaxPooling1D(2),
    # tf.keras.layers.Conv1D(64, 3, activation='relu'),
    # tf.keras.layers.MaxPooling1D(2),
    # tf.keras.layers.Conv1D(64, 3, activation='relu'),
    # tf.keras.layers.Conv1D(64, 3, activation='relu'),
    # tf.keras.layers.MaxPooling1D(2),


    tf.keras.layers.Flatten(input_shape=(42,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.summary()

cnn_history = model.fit(cnn_training, epochs=100, steps_per_epoch=x_train.shape[0]/64)
cnn_predictions = model.predict(cnn_testing).flatten()
print(cnn_predictions)
print(y_test)

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


del model