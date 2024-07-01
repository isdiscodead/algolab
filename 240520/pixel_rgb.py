import cv2
import os
import matplotlib.pyplot as plt

# import video file
folder_path = './ansung'
files = os.listdir(folder_path)

frame_cnt = len(files)

for file in files:
    # read video file frame by frame
    file_path = os.path.join(folder_path, file)
    image = cv2.imread(file_path)

    # BGR to RGB
    B, G, R = cv2.split(image)

    # dist of pixels
    hist_R = cv2.calcHist([R], [0], None, [256], [0, 256])
    hist_G = cv2.calcHist([G], [0], None, [256], [0, 256])
    hist_B = cv2.calcHist([B], [0], None, [256], [0, 256])

    # plot
    plt.plot(hist_R, color='red')
    plt.plot(hist_G, color='green')
    plt.plot(hist_B, color='blue')

    plt.show()

# close object and window
cv2.destroyAllWindows()