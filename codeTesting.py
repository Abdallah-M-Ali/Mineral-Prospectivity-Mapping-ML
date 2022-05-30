from Data_preprocessing import rs_preprocessing, dataFitting

landsat = 'D:/programes/dataset/aster-finalstack2.tif'
band_data1, img_as_array1 = rs_preprocessing(landsat, reshape=True)

train_ds = 'D:/programes/qgis/train_reg.shp'
x_train, y_train = dataFitting(landsat, band_data1, train_ds)
print(x_train)
print(y_train)