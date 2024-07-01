import cv2
import os
import matplotlib.pyplot as plt

# import video file
folder_path = './jochiwon'
files = os.listdir(folder_path)

frame_cnt = len(files)

for file in files:
    # read video file frame by frame
    file_path = os.path.join(folder_path, file)
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # calc hist
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # plot
    plt.subplot(1, 2, 1)
    plt.plot(hist, color='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')

    plt.show()

# close object and window
cv2.destroyAllWindows()