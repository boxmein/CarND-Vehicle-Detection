#!/usr/bin/env python3
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import sys
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

from lesson_functions import *

color_space = 'YCrCb'        # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9                   # HOG orientations
pix_per_cell = 8             # HOG pixels per cell
cell_per_block = 2           # HOG cells per block
hog_channel = "ALL"          # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)      # Spatial binning dimensions
hist_bins = 8                # Number of histogram bins
spatial_feat = True          # Spatial features on or off
hist_feat = True             # Histogram features on or off
hog_feat = True              # HOG features on or off

C = 1.0                      # C value for SVM

y_start_stop_xl = [520, 720] # Min/max for XLarge
y_start_stop_lg = [460, 720] # Min and max Large
y_start_stop_md = [399, 560] # Min and max Medium
y_start_stop_sm = [430, 470] # Min and max Smol

WS = 32                      # Window size, px

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

cars, notcars = read_test_images()
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

image = mpimg.imread('data/bbox-example-image.jpg')

heat = np.zeros_like(image[:,:,0]).astype(np.float)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
#image = image.astype(np.float32)/255

def find_hot_windows(image, y_start_stop, xy_window, xy_overlap):
    windows = slide_window(image, x_start_stop=[None,None], y_start_stop=y_start_stop, xy_window=xy_window, xy_overlap=xy_overlap)
    hot_win = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
    return windows, hot_win

def run_image(image):
    global heat
    
    win_xl, hot_xl = find_hot_windows(image, y_start_stop_xl, (WS * 9, WS * 9), (0.75, 0.75))
    win_lg, hot_lg = find_hot_windows(image, y_start_stop_lg, (WS * 6, WS * 6), (0.75, 0.75))
    win_md, hot_md = find_hot_windows(image, y_start_stop_md, (WS * 3, WS * 3), (0.5, 0.5))
    win_sm, hot_sm = find_hot_windows(image, y_start_stop_sm, (WS    , WS    ), (0.5, 0.5))
    
    # Draw Hot windows as red
    window_img = np.copy(image)
    window_img = draw_boxes(window_img, hot_xl, color=(255, 0, 0), thick=6)
    window_img = draw_boxes(window_img, hot_lg, color=(255, 0, 0), thick=6)
    window_img = draw_boxes(window_img, hot_md, color=(255, 0, 0), thick=6)
    window_img = draw_boxes(window_img, hot_sm, color=(255, 0, 0), thick=6)

    # draw all windows
    window_img = draw_boxes(window_img, win_xl, color=(255,255,0), thick=1)
    window_img = draw_boxes(window_img, win_lg, color=(0,255,0), thick=1)
    window_img = draw_boxes(window_img, win_md, color=(0,0,255), thick=1)
    window_img = draw_boxes(window_img, win_sm, color=(0,255,255), thick=1)
    
    # heat *= 0.8

    # Add heat to each box in box list
    heat = add_heat(heat, hot_xl)
    heat = add_heat(heat, hot_lg)
    heat = add_heat(heat, hot_md)
    heat = add_heat(heat, hot_sm)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return window_img, heatmap, draw_img


# In[10]:

# window_img, heatmap, draw_img = run_image(image)
# f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(30, 15))
# ax1.imshow(window_img)
# ax2.imshow(heatmap, cmap='gray')
# ax3.imshow(draw_img)
# plt.show()

def run_vid(img):
    window_img, heatmap, draw_img = run_image(img)
    cv2.imshow("Windows", window_img)
    cv2.imshow("Heatmap", heatmap)
    
    hxmap = np.dstack( (heatmap, heatmap, heatmap) ) * 30
    cv2.imshow("Drawn", draw_img)
    cv2.waitKey(3)
    return np.vstack( (window_img, hxmap, draw_img) )

vid_name = sys.argv[1] if len(sys.argv) > 1 else "project_video.mp4"

clip = VideoFileClip(vid_name)
output = clip.fl_image(run_vid)
output.write_videofile("output_images/" + str(time.time()) + "_" + vid_name, audio=False)
