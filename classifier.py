from sklearn.svm import LinearSVC #
from sklearn.preprocessing import StandardScaler #
from sklearn.model_selection import train_test_split #
import glob #
import time #
import matplotlib.image as mpimg #
import numpy as np
import cv2
import pickle

from lesson_functions import *
from hyperparameters import *


def read_test_images():
    test_images = glob.glob('data/vehicle_or_not/**/*.jpeg')
    test_cars = []
    test_notcars = []
    for image in test_images:
        if 'image' in image or 'extra' in image:
            test_notcars.append(image)
        else:
            test_cars.append(image)
    return test_cars, test_notcars

def read_images():
    cars = glob.glob('data/vehicles/**/*.png')
    notcars = glob.glob('data/non-vehicles/**/*.png')
    return cars, notcars

cars, notcars = read_images()
print("Cars:", len(cars))
print("Not cars:", len(notcars))

car_features = extract_features(cars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)


# In[6]:

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
print("y: ", y.shape)
print("X: ", scaled_X.shape)
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)


# In[7]:

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

with open("classified.p", "wb") as cf:
    pickle.dump({
        "svc": svc,
        "X_scaler": X_scaler
    }, cf)