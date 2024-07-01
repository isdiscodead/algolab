import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# import video file
folder_path = './jochiwon'
files = os.listdir(folder_path)

frame_cnt = len(files)

for file in files:
    # read video file frame by frame
    file_path = os.path.join(folder_path, file)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # to 7bit
    gray_7bit = np.right_shift(image, 1)

    # calc hist
    hist = cv2.calcHist([gray_7bit], [0], None, [256], [0, 256])

    # plot
    plt.subplot(1, 2, 1)
    plt.plot(hist, color='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(gray_7bit, cmap='gray')

    plt.show()

# close object and window
cv2.destroyAllWindows()