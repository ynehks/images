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
csv_file = open("report.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow([
    "Image",
    "Objects",
    "Round Objects",
    "Rectangle Objects"
])

for image_name in images:
    print(f"Обработка: {image_name}")
    path = os.path.join(INPUT_FOLDER, image_name)
    image = cv2.imread(path)
    if image is None:
        print("Ошибка загрузки")
        continue
    image = cv2.resize(image, (800, 600))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    result = image.copy()
    count = 0
    round_objects = 0
    rectangle_objects = 0
