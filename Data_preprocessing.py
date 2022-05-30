# This the basic code for remote sensing data preprocessing, where the stacked data is opened and prepared as feature
# predictors (x, x_test). the samples of ore deposit are processed as target variables (y, y_test)
# The code also contain how data is preprocessed to fit different ML models requirement.
# The code is mainly several function which can be called in others python codes within the same directory

from osgeo import gdal
from osgeo import ogr
import tensorflow as tf
import numpy as np
import geopandas as gpd


driverTiff = gdal.GetDriverByName('GTiff')





# this function for opening the stacked data to extract the x_train and x_test
# as well and reshaping the image to become as np array
def rs_preprocessing (data, reshape=True):
    rs_ds = gdal.Open(data)
    nbands = rs_ds.RasterCount
    band_data = []
    print('bands', rs_ds.RasterCount, 'rows', rs_ds.RasterYSize, 'columns',
          rs_ds.RasterXSize)
    for i in range(1, nbands + 1):
        band = rs_ds.GetRasterBand(i).ReadAsArray()
        band_data.append(band)
    band_data = np.dstack(band_data)
    print(band_data.shape)

    if reshape == True:
        new_shape = (band_data.shape[0] * band_data.shape[1], band_data.shape[2])
        img_as_array = band_data[:, :, :np.int(band_data.shape[2])].reshape(new_shape)
        print('Reshaped from {o} to {n}'.format(o=band_data.shape, n=img_as_array.shape))
        img_as_array = np.nan_to_num(img_as_array)
        return band_data, img_as_array
    else:
        return band_data





# the next function accept shapefile data contains points as target variables samples
# the target value must be in attribute called value and values are binary (0,1)
# 1 represents that the target mineral exist and vice versa
# the function splits the data into train and test datasets and save them separately in two files

def target_variable (data, trainDirectory, testDirectory, tarinPercent=0.8):
    gdf = gpd.read_file(data)
    gdf['raster'] = np.where(gdf['value'] == 0, 1,2)
    gdf_train = gdf.sample(frac=tarinPercent)
    gdf_test = gdf.drop(gdf_train.index)
    print('gdf shape', gdf.shape, 'training', gdf_train.shape, 'test', gdf_test.shape)
    gdf_train.to_file(trainDirectory)
    gdf_test.to_file(testDirectory)
    

# Data rasterization and extraction of training and testing dataset
# The folowing function accept
# (1) The RS data directory
# (2) The processed RS data
# (3) the directory of the training or testing dataset
# function return x (variable features) and y (target variables)
def dataFitting (RSData, band_data, SHfile):
    RS_ds = gdal.Open(RSData)
    train_ds = ogr.Open(SHfile)
    lyr = train_ds.GetLayer()
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', RS_ds.RasterXSize, RS_ds.RasterYSize, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(RS_ds.GetGeoTransform())
    target_ds.SetProjection(RS_ds.GetProjectionRef())
    options = ['ATTRIBUTE=raster']
    gdal.RasterizeLayer(target_ds, [1], lyr, options=options)
    data = target_ds.GetRasterBand(1).ReadAsArray()
    print('min', data.min(), 'max', data.max(), 'mean', data.mean())
    truth = target_ds.GetRasterBand(1).ReadAsArray()
    classes = np.unique(truth)[1:]
    print('class values', classes)
    n_samples = (data > 0).sum()
    print('{n} training samples'.format(n=n_samples))
    idx = np.nonzero(truth)
    x = band_data[idx]
    y = truth[idx] - 1
    print('Our X matrix is sized: {sz}'.format(sz=x.shape))
    print('Our y array is sized: {sz}'.format(sz=np.shape(y)))

    return x, y



def tfPipline (feature, label, shuffle=True, repeat=False, BUFFER_SIZE=10000, BATCH_SIZE=64):
    if label == '':
        dataset = tf.data.Dataset.from_tensor_slices(feature)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((feature, label))
    if repeat==True:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    else:
        dataset = dataset.batch(BATCH_SIZE)

    return dataset

def main():
    print("This is the main code to test above functions")

    landsat = 'D:/programes/dataset/aster-finalstack2.tif'
    band_data1, img_as_array1 = rs_preprocessing(landsat, reshape=True)

    train_ds = 'D:/programes/qgis/train_reg.shp'
    x_train, y_train = dataFitting(landsat, band_data1, train_ds)
    print(x_train)
    print(y_train)

if __name__ == '__main__':
    main()


