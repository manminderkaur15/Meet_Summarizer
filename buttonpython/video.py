import cv2
import numpy as np
import os

image_folder = 'media'
video_file = 'meet-recording.mp4'
image_size = (160, 120)
fps = 24

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort()

out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'MP4V'), fps, image_size)

img_array = []
for filename in images:
    img = cv2.imread(os.path.join(image_folder, filename))
    img_array.append(img)
    out.write(img)

out.release()