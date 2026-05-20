import cv2
import math
import os
import csv

INPUT_FOLDER = "images"
OUTPUT_FOLDER = "results"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
images = []
for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".jpg") or \
       file.endswith(".png") or \
       file.endswith(".jpeg"):
        images.append(file)
if len(images) == 0:
    print("Изображения не найдены")
    exit()
