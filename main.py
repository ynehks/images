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
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        count += 1
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        (cx_circle, cy_circle), radius = \
            cv2.minEnclosingCircle(cnt)
        cx = x + w // 2
        cy = y + h // 2
        if perimeter != 0:
            roundness = \
                4 * math.pi * area / (perimeter ** 2)
        else:
            roundness = 0
        if roundness > 0.8:
            color = (0, 255, 0)
            round_objects += 1
            shape = "Circle"
        else:
            color = (0, 0, 255)
            rectangle_objects += 1
            shape = "Rectangle"
        cv2.drawContours(result, [cnt], -1, color, 2)
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            color,
            2
        )
        cv2.circle(
            result,
            (int(cx_circle), int(cy_circle)),
            int(radius),
            (255, 255, 0),
            1
        )
        cv2.circle(
            result,
            (cx, cy),
            4,
            (255, 0, 255),
            -1
        )
        cv2.putText(
            result,
            str(count),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        cv2.putText(
            result,
            shape,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
        cv2.putText(
            result,
            f"S={int(area)}",
            (x, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    cv2.putText(
        result,
        f"Objects: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )
    cv2.putText(
        result,
        f"Round: {round_objects}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    cv2.putText(
        result,
        f"Rectangle: {rectangle_objects}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )
    save_path = os.path.join(
        OUTPUT_FOLDER,
        f"result_{image_name}"
    )
    cv2.imwrite(save_path, result)
    writer.writerow([
        image_name,
        count,
        round_objects,
        rectangle_objects
    ])
    cv2.imshow("Result", result)
    key = cv2.waitKey(300)
    if key == 27:
        break
csv_file.close()
cv2.destroyAllWindows()
print("Обработка завершена")
