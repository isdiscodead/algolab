import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# import folder
folder_path = './ansung'
files = os.listdir(folder_path)

frame_cnt = len(files)

hist_acc = np.zeros([256, 1])

for file in files:
    # read file
    file_path = os.path.join(folder_path, file)
    image = cv2.imread(file_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # dist of pixels
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_acc += hist


# close object and window
cv2.destroyAllWindows()

# plot
plt.xlabel("Pixel Value")
plt.xlim([0, 256])
plt.plot(hist_acc / frame_cnt)

plt.show()