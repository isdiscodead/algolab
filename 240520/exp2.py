import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# import video file
folder_path = './jochiwon'
files = os.listdir(folder_path)

frame_cnt = len(files)

# filters
sharpening_mask = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

for file in files:
    # read video file frame by frame
    file_path = os.path.join(folder_path, file)
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # add filter
    sharpening_out = cv2.filter2D(image, -1, sharpening_mask)

    # clahe
    clahe = cv2.createCLAHE(clipLimit=160, tileGridSize=(2,2))
    clahe_img = clahe.apply(sharpening_out)

    # calc hist
    hist = cv2.calcHist([clahe_img], [0], None, [256], [0, 256])

    # plot
    plt.subplot(1, 2, 1)
    plt.plot(hist, color='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(clahe_img, cmap='gray')
    

    plt.show()

# close object and window
cv2.destroyAllWindows()