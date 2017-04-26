color_space = 'YCrCb'        # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9                   # HOG orientations
pix_per_cell = 8             # HOG pixels per cell
cell_per_block = 2           # HOG cells per block
hog_channel = "ALL"          # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)      # Spatial binning dimensions
hist_bins = 32               # Number of histogram bins
spatial_feat = True          # Spatial features on or off
hist_feat = True             # Histogram features on or off
hog_feat = True              # HOG features on or off

C = 0.1                      # C value for SVM

y_start_stop_xl = [520, 720] # Min/max for XLarge
y_start_stop_lg = [460, 720] # Min and max Large
y_start_stop_md = [399, 560] # Min and max Medium
y_start_stop_sm = [399, 450] # Min and max Smol

x_start_stop_xl = [None, None] # Min/max for XLarge
x_start_stop_lg = [None, None] # Min and max Large
x_start_stop_md = [450, 1280] # Min and max Medium
x_start_stop_sm = [520, 930] # Min and max Smol

WS = 32                      # Window size, px
