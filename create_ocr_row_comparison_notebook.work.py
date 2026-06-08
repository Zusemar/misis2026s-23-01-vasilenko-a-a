from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "ocr_rows_tesseract_easyocr_ssocr.ipynb"


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest()[:8],
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": hashlib.sha1(source.encode("utf-8")).hexdigest()[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown_cell(
        """# OCR строк семисегментного дисплея

Ноутбук сравнивает четыре распознавателя на строках цифр:

- **Tesseract**;
- **EasyOCR**;
- **ssocr** — специализированный OCR семисегментных дисплеев;
- **Pic7Seg** — существующий алгоритм из `pic.ipynb`, используемый как baseline.

До выделения строк алгоритм автоматически нормализует ориентацию без
использования разметки:

- проверяет повороты `0°`, `90°`, `180°`, `270°`;
- выбирает вариант с наиболее правдоподобной структурой строк цифр;
- исправляет небольшой остаточный наклон по центрам найденных цифр.

После этого строки выделяются существующим пайплайном из `pic.ipynb`:

1. итоговая бинарная маска из `best_postprocessed`;
2. дилатация прямоугольным ядром `7×7`, две итерации;
3. дополнительная эрозия;
4. поиск контуров, фильтр `fill_ratio`;
5. группировка цифр по строкам `AgglomerativeClustering`;
6. вырезание общей области строки.

Разметка `detection_etalon` используется **только после работы алгоритма** для
сопоставления результата и расчёта метрик. Поддерживаются `rectangle`,
`oriented_rectangle` и `polygon`.

## Метрики

- `ExactRowAccuracy` — доля строк, распознанных полностью правильно;
- `TotalCER` — суммарное расстояние Левенштейна / суммарное число GT-символов;
- `MeanCER` — среднее CER по строкам;
- `NonEmptyRate` — доля строк, где OCR вернул хотя бы одну цифру;
- `LengthAccuracy` — доля строк с правильным количеством цифр.
"""
    ),
    code_cell(
        """import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
import torch
from IPython.display import display
from sklearn.cluster import AgglomerativeClustering
from tqdm.auto import tqdm

ROOT = Path.cwd()
BINARY_DIR = ROOT / "best_postprocessed"
ETALON_DIR = ROOT / "detection_etalon"
RESULT_DIR = ROOT / "ocr_row_results"
CROP_DIR = RESULT_DIR / "crops"
if RESULT_DIR.exists():
    shutil.rmtree(RESULT_DIR)
RESULT_DIR.mkdir(exist_ok=True)
CROP_DIR.mkdir(exist_ok=True)
REUSE_OCR_RESULTS = False

TESSERACT_CMD = shutil.which("tesseract")
SSOCR_CMD = shutil.which("ssocr")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

print("Python:", sys.version.split()[0])
print("OpenCV:", cv2.__version__)
print("Tesseract:", TESSERACT_CMD)
print("EasyOCR:", easyocr.__version__)
print("PyTorch:", torch.__version__)
print("ssocr:", SSOCR_CMD)
assert TESSERACT_CMD, "Tesseract не найден"
assert SSOCR_CMD, "ssocr не найден"
"""
    ),
    markdown_cell(
        """## Загрузка разметки строк

Правильная строка строится из `description` всех цифр с одинаковым `label` в
порядке аннотаций. Пустая `description` означает неполную разметку: такая
строка показывается, но исключается из текстовых метрик.
"""
    ),
    code_cell(
        """ROW_ORDER = {
    "first row": 0,
    "second row": 1,
    "third row": 2,
    "fourth row": 3,
    "fifth row": 4,
}


def normalize_key(path):
    stem = Path(path).stem
    if stem.endswith("-1"):
        stem = stem[:-2]
    return stem.replace("_rectified", "")


def points_to_box(points):
    array = np.asarray(points, dtype=float)
    return (
        int(np.floor(array[:, 0].min())),
        int(np.floor(array[:, 1].min())),
        int(np.ceil(array[:, 0].max())),
        int(np.ceil(array[:, 1].max())),
    )


def union_box(boxes):
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def rotate_bound_with_matrix(image, angle, background=0):
    image_h, image_w = image.shape[:2]
    center = (image_w / 2, image_h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_w = int(np.ceil(image_h * sine + image_w * cosine))
    new_h = int(np.ceil(image_h * cosine + image_w * sine))
    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]
    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderValue=background,
    )
    return rotated, matrix


def compose_affine(second, first):
    first_h = np.vstack([first, [0, 0, 1]])
    second_h = np.vstack([second, [0, 0, 1]])
    return (second_h @ first_h)[:2]


def transform_points(points, matrix):
    points = np.asarray(points, dtype=float)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return homogeneous @ matrix.T


def load_gt_rows(json_path):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for annotation_index, shape in enumerate(data.get("shapes", [])):
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        label = shape.get("label", "").strip().lower()
        raw_digit = str(shape.get("description", "")).strip()
        digit = "".join(re.findall(r"\\d", raw_digit))
        grouped[label].append(
            {
                "digit": digit,
                "raw_digit": raw_digit,
                "box": points_to_box(points),
                "center": tuple(np.asarray(points, dtype=float).mean(axis=0)),
                "points": np.asarray(points, dtype=float),
                "annotation_index": annotation_index,
                "shape_type": shape.get("shape_type", ""),
            }
        )

    rows = []
    for label, digits in grouped.items():
        digits.sort(key=lambda item: item["annotation_index"])
        x_centers = [item["center"][0] for item in digits]
        y_centers = [item["center"][1] for item in digits]
        vertical = (max(y_centers) - min(y_centers)) > (max(x_centers) - min(x_centers))
        text = "".join(item["digit"] if item["digit"] else "?" for item in digits)
        rows.append(
            {
                "label": label,
                "text": text,
                "valid_text": all(bool(item["digit"]) for item in digits),
                "vertical": vertical,
                "box": union_box([item["box"] for item in digits]),
                "digits": digits,
            }
        )
    rows.sort(
        key=lambda row: (
            ROW_ORDER.get(row["label"], 100),
            (row["box"][1] + row["box"][3]) / 2,
        )
    )
    return rows


def transform_gt_rows(rows, matrix):
    transformed_rows = []
    for row in rows:
        transformed_digits = []
        for digit in row["digits"]:
            points = transform_points(digit["points"], matrix)
            transformed_digits.append(
                {
                    **digit,
                    "points": points,
                    "center": tuple(points.mean(axis=0)),
                    "box": points_to_box(points),
                }
            )
        transformed_rows.append(
            {
                **row,
                "digits": transformed_digits,
                "box": union_box([digit["box"] for digit in transformed_digits]),
            }
        )
    return transformed_rows


binary_by_key = {normalize_key(path): path for path in BINARY_DIR.glob("*.png")}
json_by_key = {normalize_key(path): path for path in ETALON_DIR.glob("*.json")}
pair_keys = sorted(binary_by_key.keys() & json_by_key.keys())

gt_summary = []
for key in pair_keys:
    for row in load_gt_rows(json_by_key[key]):
        gt_summary.append(
            {
                "ImageKey": key,
                "Row": row["label"],
                "GT": row["text"],
                "ValidText": row["valid_text"],
            }
        )
gt_summary = pd.DataFrame(gt_summary)

print(f"Бинарных изображений: {len(binary_by_key)}")
print(f"JSON-разметок: {len(json_by_key)}")
print(f"Совпавших пар: {len(pair_keys)}")
print(f"Всего размеченных строк: {len(gt_summary)}")
print(f"Строк с полным текстом: {gt_summary['ValidText'].sum()}")
display(gt_summary[~gt_summary["ValidText"]])
"""
    ),
    markdown_cell(
        """## Пайплайн выделения строк из `pic.ipynb`

Ниже воспроизводится существующая морфология, контурная детекция,
`fill_ratio`, агломеративная группировка и фильтр перекрывающихся боксов.
"""
    ),
    code_cell(
        """MIN_HEIGHT_RATIO = 0.05
MAX_HEIGHT_RATIO = 0.30
MIN_AREA_RATIO = 0.005
MAX_AREA_RATIO = 0.50


def apply_pic_morphology(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    result = cv2.dilate(binary, kernel, iterations=2)
    return cv2.erode(result, kernel, iterations=1)


def detect_boxes(binary, min_size=5):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_size and h >= min_size:
            boxes.append((x, y, x + w, y + h))
    return boxes


def fill_ratio_filter(box, image_h, image_w, binary):
    x1, y1, x2, y2 = box
    box_h = y2 - y1
    box_w = x2 - x1
    area = box_w * box_h
    image_area = image_h * image_w
    height_ratio = box_h / image_h
    area_ratio = area / image_area
    if not (
        MIN_HEIGHT_RATIO <= height_ratio <= MAX_HEIGHT_RATIO
        and MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO
    ):
        return False
    roi = binary[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    fill = np.sum(roi > 0) / roi.size
    return 0.1 <= fill <= 0.8


def group_agglomerative(boxes, image_h, distance_threshold=0.1):
    if len(boxes) < 2:
        return [boxes] if boxes else []
    centers_y = np.array(
        [(y1 + y2) / 2 for x1, y1, x2, y2 in boxes]
    ).reshape(-1, 1) / image_h
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
    ).fit_predict(centers_y)
    groups = [[] for _ in range(max(labels) + 1)]
    for box, label in zip(boxes, labels):
        groups[label].append(box)
    groups.sort(key=lambda group: np.mean([(box[1] + box[3]) / 2 for box in group]))
    for group in groups:
        group.sort(key=lambda box: (box[0] + box[2]) / 2)
    return groups


def filter_row_overlaps(groups):
    filtered = []
    for group in groups:
        group = sorted(group, key=lambda box: box[0])
        remove = set()
        for index in range(len(group) - 1):
            first, second = group[index], group[index + 1]
            first_h = first[3] - first[1]
            second_h = second[3] - second[1]
            if abs(first_h - second_h) / max(first_h, second_h) < 0.2:
                overlap = max(0, min(first[2], second[2]) - max(first[0], second[0]))
                min_width = min(first[2] - first[0], second[2] - second[0])
                if overlap > 0.3 * min_width:
                    remove.update([index, index + 1])
        clean_group = [box for index, box in enumerate(group) if index not in remove]
        if clean_group:
            filtered.append(clean_group)
    return filtered


def crop_box(binary, box, margin_ratio=0.12, min_margin=8):
    image_h, image_w = binary.shape
    x1, y1, x2, y2 = box
    margin = max(min_margin, int((y2 - y1) * margin_ratio))
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(image_w, x2 + margin)
    y2 = min(image_h, y2 + margin)
    return binary[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def extract_predicted_rows(binary):
    cleaned = apply_pic_morphology(binary)
    image_h, image_w = cleaned.shape
    raw_boxes = detect_boxes(cleaned)
    boxes = [
        box for box in raw_boxes
        if fill_ratio_filter(box, image_h, image_w, cleaned)
    ]
    groups = filter_row_overlaps(group_agglomerative(boxes, image_h))
    rows = []
    for group in groups:
        box = union_box(group)
        crop, crop_coordinates = crop_box(cleaned, box)
        rows.append(
            {
                "box": box,
                "crop_box": crop_coordinates,
                "crop": crop,
                "digit_boxes": group,
            }
        )
    rows.sort(key=lambda row: (row["box"][1] + row["box"][3]) / 2)
    return cleaned, rows


def orientation_structure_score(cleaned, rows):
    if not rows:
        return -1000.0

    image_h, image_w = cleaned.shape
    row_count = len(rows)
    score = 5.0 - 2.0 * abs(row_count - 3)
    all_boxes = [box for row in rows for box in row["digit_boxes"]]

    for row in rows:
        boxes = row["digit_boxes"]
        count = len(boxes)
        score += 2.0 if 2 <= count <= 4 else -1.5 * abs(count - 3)

        heights = np.array([box[3] - box[1] for box in boxes], dtype=float)
        widths = np.array([box[2] - box[0] for box in boxes], dtype=float)
        portrait_ratio = np.mean(heights / np.maximum(widths, 1))
        score += float(np.clip(portrait_ratio, 0, 3))

        centers_y = np.array([(box[1] + box[3]) / 2 for box in boxes])
        score -= float(np.std(centers_y) / max(np.median(heights), 1)) * 3

    if len(rows) >= 2:
        row_heights = [
            np.median([box[3] - box[1] for box in row["digit_boxes"]])
            for row in rows
        ]
        score += float((row_heights[0] - row_heights[-1]) / max(image_h, 1)) * 6
        score += 0.35 * (len(rows[0]["digit_boxes"]) - len(rows[-1]["digit_boxes"]))

    foreground_ratio = np.count_nonzero(cleaned) / cleaned.size
    if 0.01 <= foreground_ratio <= 0.45:
        score += 1.0
    score += min(len(all_boxes), 12) * 0.15
    return float(score)


def estimate_residual_skew(rows):
    angles = []
    for row in rows:
        boxes = row["digit_boxes"]
        if len(boxes) < 2:
            continue
        first = boxes[0]
        last = boxes[-1]
        first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
        last_center = ((last[0] + last[2]) / 2, (last[1] + last[3]) / 2)
        angles.append(
            np.degrees(
                np.arctan2(
                    last_center[1] - first_center[1],
                    last_center[0] - first_center[0],
                )
            )
        )
    if not angles:
        return 0.0
    angle = float(np.median(angles))
    return angle if abs(angle) <= 20 else 0.0


def normalize_orientation(binary):
    candidates = []
    for angle in [0, 90, 180, 270]:
        rotated, matrix = rotate_bound_with_matrix(binary, angle)
        cleaned, rows = extract_predicted_rows(rotated)
        candidates.append(
            {
                "RightAngle": angle,
                "Matrix": matrix,
                "Binary": rotated,
                "Cleaned": cleaned,
                "Rows": rows,
                "StructureScore": orientation_structure_score(cleaned, rows),
            }
        )

    best = max(candidates, key=lambda item: item["StructureScore"])
    residual = estimate_residual_skew(best["Rows"])
    if abs(residual) >= 0.5:
        deskewed, deskew_matrix = rotate_bound_with_matrix(best["Binary"], residual)
        cleaned, rows = extract_predicted_rows(deskewed)
        deskew_score = orientation_structure_score(cleaned, rows)
        if deskew_score >= best["StructureScore"] - 0.5:
            best = {
                **best,
                "Matrix": compose_affine(deskew_matrix, best["Matrix"]),
                "Binary": deskewed,
                "Cleaned": cleaned,
                "Rows": rows,
                "StructureScore": deskew_score,
                "ResidualAngle": residual,
            }
    best.setdefault("ResidualAngle", 0.0)
    return best, candidates


def vertical_iou(first, second):
    intersection = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    union = max(first[3], second[3]) - min(first[1], second[1])
    return intersection / union if union else 0.0


def match_rows(gt_rows, predicted_rows, threshold=0.1):
    candidates = []
    for gt_index, gt_row in enumerate(gt_rows):
        for pred_index, pred_row in enumerate(predicted_rows):
            candidates.append(
                (vertical_iou(gt_row["box"], pred_row["box"]), gt_index, pred_index)
            )
    candidates.sort(reverse=True)
    matches = {}
    used_predictions = set()
    for score, gt_index, pred_index in candidates:
        if score < threshold:
            continue
        if gt_index not in matches and pred_index not in used_predictions:
            matches[gt_index] = pred_index
            used_predictions.add(pred_index)
    return matches
"""
    ),
    markdown_cell(
        """## Baseline Pic7Seg

Это распознаватель семи сегментов из `pic.ipynb`. Он применяется к отдельным
детектированным боксам цифр, после чего цифры объединяются в строку.
"""
    ),
    code_cell(
        """def recognize_pic7seg_digit(roi):
    if roi is None or roi.size == 0:
        return "?"
    image_h, image_w = roi.shape
    border_mean = (
        np.mean(roi[0, :])
        + np.mean(roi[-1, :])
        + np.mean(roi[:, 0])
        + np.mean(roi[:, -1])
    ) / 4
    if border_mean > 127:
        roi = 255 - roi
    roi = cv2.medianBlur(roi, 3)
    if image_w / image_h < 0.5:
        return "1"

    coordinates = {
        "A": (0.2, 0.0, 0.8, 0.3),
        "B": (0.7, 0.2, 1.0, 0.5),
        "C": (0.7, 0.5, 1.0, 0.8),
        "D": (0.2, 0.7, 0.8, 1.0),
        "E": (0.0, 0.5, 0.3, 0.8),
        "F": (0.0, 0.2, 0.3, 0.5),
        "G": (0.2, 0.4, 0.8, 0.6),
    }
    active = {}
    for segment, (x1r, y1r, x2r, y2r) in coordinates.items():
        x1, y1 = int(x1r * image_w), int(y1r * image_h)
        x2, y2 = int(x2r * image_w), int(y2r * image_h)
        segment_roi = roi[y1:y2, x1:x2]
        active[segment] = (
            segment_roi.size > 0
            and np.sum(segment_roi > 128) / segment_roi.size > 0.3
        )

    digit_segments = {
        "0": ("A", "B", "C", "D", "E", "F"),
        "1": ("B", "C"),
        "2": ("A", "B", "D", "E", "G"),
        "3": ("A", "B", "C", "D", "G"),
        "4": ("B", "C", "F", "G"),
        "5": ("A", "C", "D", "F", "G"),
        "6": ("A", "C", "D", "E", "F", "G"),
        "7": ("A", "B", "C"),
        "8": ("A", "B", "C", "D", "E", "F", "G"),
        "9": ("A", "B", "C", "D", "F", "G"),
    }
    best_digit, best_score = "?", -1
    for digit, required in digit_segments.items():
        match = sum(active.get(segment, False) for segment in required)
        extra = sum(
            is_active
            for segment, is_active in active.items()
            if is_active and segment not in required
        )
        score = match - 0.5 * extra
        if score > best_score:
            best_digit, best_score = digit, score
    return best_digit


def recognize_pic7seg_row(cleaned, predicted_row):
    digits = []
    for x1, y1, x2, y2 in predicted_row["digit_boxes"]:
        digits.append(recognize_pic7seg_digit(cleaned[y1:y2, x1:x2]))
    return "".join(digits)
"""
    ),
    markdown_cell(
        """## Формирование датасета вырезанных строк

В каждой записи хранится GT-строка, найденная область строки и результат
существующего `Pic7Seg`. Если строка не была найдена, OCR получает пустой ответ:
это учитывается как end-to-end ошибка.
"""
    ),
    code_cell(
        """row_records = []
extraction_records = []
orientation_records = []
row_images = {}

for key in tqdm(pair_keys, desc="Выделение строк"):
    binary = cv2.imread(str(binary_by_key[key]), cv2.IMREAD_GRAYSCALE)
    orientation, candidates = normalize_orientation(binary)
    cleaned = orientation["Cleaned"]
    predicted_rows = orientation["Rows"]
    gt_rows = transform_gt_rows(load_gt_rows(json_by_key[key]), orientation["Matrix"])
    matches = match_rows(gt_rows, predicted_rows)

    orientation_records.append(
        {
            "ImageKey": key,
            "RightAngle": orientation["RightAngle"],
            "ResidualAngle": orientation["ResidualAngle"],
            "StructureScore": orientation["StructureScore"],
            **{
                f"Score{candidate['RightAngle']}": candidate["StructureScore"]
                for candidate in candidates
            },
        }
    )
    extraction_records.append(
        {
            "ImageKey": key,
            "RightAngle": orientation["RightAngle"],
            "ResidualAngle": orientation["ResidualAngle"],
            "GTRows": len(gt_rows),
            "PredictedRows": len(predicted_rows),
            "MatchedRows": len(matches),
        }
    )

    for gt_index, gt_row in enumerate(gt_rows):
        predicted_row = predicted_rows[matches[gt_index]] if gt_index in matches else None
        record_id = f"{key}__{gt_row['label'].replace(' ', '_')}"
        pic_prediction = ""
        if predicted_row is not None:
            row_images[record_id] = predicted_row["crop"]
            cv2.imwrite(str(CROP_DIR / f"{record_id}.png"), predicted_row["crop"])
            pic_prediction = recognize_pic7seg_row(cleaned, predicted_row)

        row_records.append(
            {
                "RecordID": record_id,
                "ImageKey": key,
                "Row": gt_row["label"],
                "GT": gt_row["text"],
                "ValidText": gt_row["valid_text"],
                "RowFound": predicted_row is not None,
                "Pic7Seg": pic_prediction,
            }
        )

rows_df = pd.DataFrame(row_records)
extraction_df = pd.DataFrame(extraction_records)
orientation_df = pd.DataFrame(orientation_records)
rows_df.to_csv(RESULT_DIR / "row_dataset.csv", index=False)
extraction_df.to_csv(RESULT_DIR / "row_extraction_metrics.csv", index=False)
orientation_df.to_csv(RESULT_DIR / "orientation_diagnostics.csv", index=False)

print(f"Всего GT-строк: {len(rows_df)}")
print(f"Полностью размеченных GT-строк: {rows_df['ValidText'].sum()}")
print(f"Найденных строк: {rows_df['RowFound'].sum()}")
print(f"Найдено изображений строк: {len(row_images)}")
display(orientation_df["RightAngle"].value_counts().sort_index().to_frame("Images"))
display(extraction_df.describe().round(3))
display(rows_df.head(10))
"""
    ),
    code_cell(
        """sample_rows = rows_df[rows_df["RowFound"]].head(12)
figure, axes = plt.subplots(3, 4, figsize=(16, 9))
for axis, (_, row) in zip(axes.ravel(), sample_rows.iterrows()):
    axis.imshow(row_images[row["RecordID"]], cmap="gray", vmin=0, vmax=255)
    axis.set_title(
        f"{row['ImageKey']} | {row['Row']}\\nGT={row['GT']} | Pic7Seg={row['Pic7Seg']}",
        fontsize=8,
    )
    axis.axis("off")
for axis in axes.ravel()[len(sample_rows):]:
    axis.axis("off")
plt.tight_layout()
plt.show()
"""
    ),
    markdown_cell(
        """## Метрики текста и подготовка изображения

Для каждого OCR перебираются масштаб и полярность. При `black_on_white`
семисегментные цифры становятся чёрными на белом фоне; при `white_on_black`
сохраняется исходная полярность итоговой маски.
"""
    ),
    code_cell(
        """def normalize_digits(value):
    return "".join(re.findall(r"\\d", str(value)))


def edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for first_index, first_char in enumerate(first, start=1):
        current = [first_index]
        for second_index, second_char in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[second_index] + 1,
                    previous[second_index - 1] + (first_char != second_char),
                )
            )
        previous = current
    return previous[-1]


def prepare_row_image(crop, polarity="black_on_white", scale=2, padding=20):
    if polarity == "black_on_white":
        prepared = 255 - crop
        background = 255
    else:
        prepared = crop.copy()
        background = 0
    if scale != 1:
        prepared = cv2.resize(
            prepared,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
    return cv2.copyMakeBorder(
        prepared,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=background,
    )


def make_detail_row(model, params, source_row, prediction):
    gt = source_row["GT"]
    prediction = normalize_digits(prediction)
    distance = edit_distance(gt, prediction)
    return {
        "Model": model,
        "Params": params,
        "RecordID": source_row["RecordID"],
        "ImageKey": source_row["ImageKey"],
        "Row": source_row["Row"],
        "GT": gt,
        "Prediction": prediction,
        "Exact": prediction == gt,
        "EditDistance": distance,
        "CER": distance / len(gt) if gt else np.nan,
        "NonEmpty": bool(prediction),
        "LengthCorrect": len(prediction) == len(gt),
        "RowFound": source_row["RowFound"],
    }


def load_detail_csv(path):
    details = pd.read_csv(
        path,
        dtype={
            "Model": str,
            "Params": str,
            "RecordID": str,
            "ImageKey": str,
            "Row": str,
            "GT": str,
            "Prediction": str,
        },
    )
    details["Prediction"] = details["Prediction"].fillna("")
    return details


def refresh_cached_metrics(details, source_rows):
    source_by_id = {
        row["RecordID"]: row
        for _, row in source_rows.iterrows()
    }
    refreshed = []
    for _, detail in details.iterrows():
        source_row = source_by_id[detail["RecordID"]]
        refreshed.append(
            make_detail_row(
                detail["Model"],
                detail["Params"],
                source_row,
                detail["Prediction"],
            )
        )
    return pd.DataFrame(refreshed)


def summarize_details(details):
    grouped = []
    for (model, params), group in details.groupby(["Model", "Params"]):
        total_gt_chars = group["GT"].str.len().sum()
        found = group[group["RowFound"]]
        found_gt_chars = found["GT"].str.len().sum()
        grouped.append(
            {
                "Model": model,
                "Params": params,
                "Rows": len(group),
                "FoundRows": len(found),
                "ExactRowAccuracy": group["Exact"].mean(),
                "TotalCER": group["EditDistance"].sum() / total_gt_chars,
                "MeanCER": group["CER"].mean(),
                "NonEmptyRate": group["NonEmpty"].mean(),
                "LengthAccuracy": group["LengthCorrect"].mean(),
                "RowFoundRate": group["RowFound"].mean(),
                "FoundOnlyExactAccuracy": found["Exact"].mean(),
                "FoundOnlyTotalCER": (
                    found["EditDistance"].sum() / found_gt_chars
                    if found_gt_chars
                    else np.nan
                ),
                "FoundOnlyNonEmptyRate": found["NonEmpty"].mean(),
                "FoundOnlyLengthAccuracy": found["LengthCorrect"].mean(),
            }
        )
    return pd.DataFrame(grouped).sort_values(
        ["FoundOnlyExactAccuracy", "FoundOnlyTotalCER", "NonEmptyRate"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


valid_rows = rows_df[rows_df["ValidText"]].copy()
print(f"Строк для вычисления OCR-метрик: {len(valid_rows)}")
"""
    ),
    markdown_cell(
        """## Pic7Seg: честный baseline на строках

В отличие от старой таблицы `recognition_results.csv`, здесь учитываются все
GT-строки, включая пропущенные детектором.
"""
    ),
    code_cell(
        """pic_details = pd.DataFrame(
    [
        make_detail_row("Pic7Seg", "pic.ipynb baseline", row, row["Pic7Seg"])
        for _, row in valid_rows.iterrows()
    ]
)
pic_summary = summarize_details(pic_details)
display(pic_summary.round(4))
"""
    ),
    markdown_cell(
        """## Tesseract: перебор параметров

Перебираются `PSM`, масштаб и полярность. Для всех вариантов установлен
whitelist `0123456789`.
"""
    ),
    code_cell(
        """TESSERACT_PSMS = [6, 7, 8, 13]
TESSERACT_SCALES = [1, 2, 3]
POLARITIES = ["black_on_white", "white_on_black"]

if REUSE_OCR_RESULTS and (RESULT_DIR / "tesseract_details.csv").exists():
    tesseract_details = refresh_cached_metrics(
        load_detail_csv(RESULT_DIR / "tesseract_details.csv"),
        valid_rows,
    )
    print("Использованы сохранённые детальные результаты Tesseract")
else:
    tesseract_rows = []
    for psm, scale, polarity in tqdm(
        list(product(TESSERACT_PSMS, TESSERACT_SCALES, POLARITIES)),
        desc="Tesseract grid",
    ):
        params = f"psm={psm};scale={scale};polarity={polarity}"
        for _, row in valid_rows.iterrows():
            prediction = ""
            if row["RowFound"]:
                prepared = prepare_row_image(row_images[row["RecordID"]], polarity, scale)
                prediction = pytesseract.image_to_string(
                    prepared,
                    config=f"--psm {psm} -c tessedit_char_whitelist=0123456789",
                )
            tesseract_rows.append(make_detail_row("Tesseract", params, row, prediction))
    tesseract_details = pd.DataFrame(tesseract_rows)

tesseract_summary = summarize_details(tesseract_details)
tesseract_details.to_csv(RESULT_DIR / "tesseract_details.csv", index=False)
tesseract_summary.to_csv(RESULT_DIR / "tesseract_summary.csv", index=False)
display(tesseract_summary.head(10).round(4))
"""
    ),
    markdown_cell(
        """## ssocr: перебор параметров

`ssocr` специализирован для семисегментных дисплеев. Перебираются порог,
минимальный размер сегмента, распознавание единицы и морфологическая команда.
"""
    ),
    code_cell(
        """SSOCR_THRESHOLDS = [30, 40, 50, 60, 70]
SSOCR_MIN_SEGMENTS = [1, 3]
SSOCR_ONE_RATIOS = [2, 3, 4]
SSOCR_OPERATIONS = ["none", "closing", "opening"]


def run_ssocr(image, threshold, min_segment, one_ratio, operation):
    encoded_ok, encoded = cv2.imencode(".png", image)
    if not encoded_ok:
        return ""
    command = [
        SSOCR_CMD,
        "-d", "-1",
        "-c", "digits",
        "-f", "black",
        "-b", "white",
        "-t", str(threshold),
        "-N", str(min_segment),
        "-r", str(one_ratio),
        "-C",
    ]
    if operation != "none":
        command.extend([operation, "1"])
    command.append("-")
    result = subprocess.run(
        command,
        input=encoded.tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.decode("utf-8", errors="ignore")


if REUSE_OCR_RESULTS and (RESULT_DIR / "ssocr_details.csv").exists():
    ssocr_details = refresh_cached_metrics(
        load_detail_csv(RESULT_DIR / "ssocr_details.csv"),
        valid_rows,
    )
    print("Использованы сохранённые детальные результаты ssocr")
else:
    ssocr_rows = []
    ssocr_grid = list(
        product(
            SSOCR_THRESHOLDS,
            SSOCR_MIN_SEGMENTS,
            SSOCR_ONE_RATIOS,
            SSOCR_OPERATIONS,
        )
    )
    for threshold, min_segment, one_ratio, operation in tqdm(ssocr_grid, desc="ssocr grid"):
        params = (
            f"threshold={threshold};min_segment={min_segment};"
            f"one_ratio={one_ratio};operation={operation}"
        )
        for _, row in valid_rows.iterrows():
            prediction = ""
            if row["RowFound"]:
                prepared = prepare_row_image(
                    row_images[row["RecordID"]],
                    polarity="black_on_white",
                    scale=2,
                )
                prediction = run_ssocr(
                    prepared,
                    threshold=threshold,
                    min_segment=min_segment,
                    one_ratio=one_ratio,
                    operation=operation,
                )
            ssocr_rows.append(make_detail_row("ssocr", params, row, prediction))
    ssocr_details = pd.DataFrame(ssocr_rows)
ssocr_summary = summarize_details(ssocr_details)
ssocr_details.to_csv(RESULT_DIR / "ssocr_details.csv", index=False)
ssocr_summary.to_csv(RESULT_DIR / "ssocr_summary.csv", index=False)
display(ssocr_summary.head(10).round(4))
"""
    ),
    markdown_cell(
        """## EasyOCR: перебор параметров

EasyOCR запускается непосредственно на уже вырезанной строке без повторной
детекции текста. Перебираются decoder, масштаб и полярность.
"""
    ),
    code_cell(
        """easy_reader = easyocr.Reader(["en"], gpu=False)
EASYOCR_DECODERS = ["greedy", "beamsearch"]
EASYOCR_SCALES = [1, 2, 3]


def run_easyocr(image, decoder):
    image_h, image_w = image.shape
    result = easy_reader.recognize(
        image,
        horizontal_list=[[0, image_w, 0, image_h]],
        free_list=[],
        decoder=decoder,
        allowlist="0123456789",
        detail=0,
        paragraph=False,
    )
    return "".join(result)


if REUSE_OCR_RESULTS and (RESULT_DIR / "easyocr_details.csv").exists():
    easyocr_details = refresh_cached_metrics(
        load_detail_csv(RESULT_DIR / "easyocr_details.csv"),
        valid_rows,
    )
    print("Использованы сохранённые детальные результаты EasyOCR")
else:
    easyocr_rows = []
    for decoder, scale, polarity in tqdm(
        list(product(EASYOCR_DECODERS, EASYOCR_SCALES, POLARITIES)),
        desc="EasyOCR grid",
    ):
        params = f"decoder={decoder};scale={scale};polarity={polarity}"
        for _, row in valid_rows.iterrows():
            prediction = ""
            if row["RowFound"]:
                prepared = prepare_row_image(row_images[row["RecordID"]], polarity, scale)
                prediction = run_easyocr(prepared, decoder)
            easyocr_rows.append(make_detail_row("EasyOCR", params, row, prediction))
    easyocr_details = pd.DataFrame(easyocr_rows)

easyocr_summary = summarize_details(easyocr_details)
easyocr_details.to_csv(RESULT_DIR / "easyocr_details.csv", index=False)
easyocr_summary.to_csv(RESULT_DIR / "easyocr_summary.csv", index=False)
display(easyocr_summary.head(10).round(4))
"""
    ),
    markdown_cell(
        """# Итоговое сравнение

Для каждого распознавателя выбирается одна конфигурация по качеству OCR на
реально найденных строках: максимальная `FoundOnlyExactAccuracy`, затем
минимальная `FoundOnlyTotalCER`. Общие end-to-end метрики по всем GT-строкам
тоже сохраняются.
"""
    ),
    code_cell(
        """all_details = pd.concat(
    [pic_details, tesseract_details, ssocr_details, easyocr_details],
    ignore_index=True,
)
all_summary = pd.concat(
    [pic_summary, tesseract_summary, ssocr_summary, easyocr_summary],
    ignore_index=True,
)
all_summary = all_summary.sort_values(
    ["FoundOnlyExactAccuracy", "FoundOnlyTotalCER", "NonEmptyRate"],
    ascending=[False, True, False],
).reset_index(drop=True)

best_by_model = (
    all_summary.sort_values(
        ["Model", "FoundOnlyExactAccuracy", "FoundOnlyTotalCER", "NonEmptyRate"],
        ascending=[True, False, True, False],
    )
    .groupby("Model", as_index=False)
    .head(1)
    .sort_values(
        ["FoundOnlyExactAccuracy", "FoundOnlyTotalCER"],
        ascending=[False, True],
    )
    .reset_index(drop=True)
)

all_details.to_csv(RESULT_DIR / "all_ocr_details.csv", index=False)
all_summary.to_csv(RESULT_DIR / "all_ocr_summary.csv", index=False)
best_by_model.to_csv(RESULT_DIR / "best_ocr_by_model.csv", index=False)

display(best_by_model.round(4))
"""
    ),
    code_cell(
        """figure, axes = plt.subplots(1, 3, figsize=(22, 6))
plot_data = best_by_model.sort_values("FoundOnlyExactAccuracy")
axes[0].barh(
    plot_data["Model"],
    plot_data["FoundOnlyExactAccuracy"],
    color="seagreen",
)
axes[0].set_xlim(0, 1)
axes[0].set_title("OCR-точность на найденных строках")
axes[0].set_xlabel("FoundOnlyExactAccuracy")
axes[0].grid(axis="x", alpha=0.25)

plot_data = best_by_model.sort_values("ExactRowAccuracy")
axes[1].barh(plot_data["Model"], plot_data["ExactRowAccuracy"], color="steelblue")
axes[1].set_xlim(0, 1)
axes[1].set_title("End-to-end точность по всем GT-строкам")
axes[1].set_xlabel("ExactRowAccuracy")
axes[1].grid(axis="x", alpha=0.25)

plot_data = best_by_model.sort_values("FoundOnlyTotalCER", ascending=False)
axes[2].barh(plot_data["Model"], plot_data["FoundOnlyTotalCER"], color="firebrick")
axes[2].set_title("OCR CER на найденных строках: меньше — лучше")
axes[2].set_xlabel("FoundOnlyTotalCER")
axes[2].grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()
"""
    ),
    markdown_cell(
        """## Сравнение со старым результатом `pic.ipynb`

Старый CSV показывает точность цифр только среди боксов, которые удалось
сопоставить с GT по IoU. Кроме того, старый загрузчик принимал только
`rectangle` и пропускал большую часть `oriented_rectangle`.

Поэтому эта цифра приведена как историческая справка и не ранжируется вместе с
end-to-end метриками строк.
"""
    ),
    code_cell(
        """old_path = ROOT / "recognition_results.csv"
if old_path.exists():
    old_results = pd.read_csv(old_path)
    old_total = old_results["total_recog"].sum()
    old_correct = old_results["correct_recog"].sum()
    old_reference = pd.DataFrame(
        [
            {
                "Metric": "Старая точность отдельных сопоставленных цифр",
                "Correct": old_correct,
                "EvaluatedDigits": old_total,
                "Accuracy": old_correct / old_total if old_total else 0,
            }
        ]
    )
    display(old_reference.round(4))
else:
    print("recognition_results.csv не найден")
"""
    ),
    markdown_cell(
        """## Визуализация ответов лучших конфигураций

Показываются строки, на которых лучшие модели расходятся. Зелёный заголовок
означает полностью правильный ответ, красный — ошибку.
"""
    ),
    code_cell(
        """best_detail_parts = []
for _, best in best_by_model.iterrows():
    part = all_details[
        (all_details["Model"] == best["Model"])
        & (all_details["Params"] == best["Params"])
    ].copy()
    best_detail_parts.append(part)
best_details = pd.concat(best_detail_parts, ignore_index=True)

pivot = best_details.pivot_table(
    index=["RecordID", "ImageKey", "Row", "GT"],
    columns="Model",
    values="Prediction",
    aggfunc="first",
).reset_index()
model_names = best_by_model["Model"].tolist()
pivot["Different"] = pivot[model_names].nunique(axis=1) > 1
pivot["AnyError"] = pivot.apply(
    lambda row: any(str(row[model]) != str(row["GT"]) for model in model_names),
    axis=1,
)
visual_rows = pivot[pivot["Different"] | pivot["AnyError"]].head(16)

figure, axes = plt.subplots(4, 4, figsize=(18, 13))
for axis, (_, row) in zip(axes.ravel(), visual_rows.iterrows()):
    axis.imshow(row_images.get(row["RecordID"], np.zeros((20, 100))), cmap="gray")
    answers = "\\n".join(
        f"{model}: {row.get(model, '')}" for model in model_names
    )
    axis.set_title(
        f"{row['ImageKey']} | {row['Row']} | GT={row['GT']}\\n{answers}",
        fontsize=7,
    )
    axis.axis("off")
for axis in axes.ravel()[len(visual_rows):]:
    axis.axis("off")
plt.tight_layout()
plt.show()
"""
    ),
    markdown_cell(
        """## Ошибки лучшего метода

Последняя таблица показывает строки с максимальным расстоянием Левенштейна для
общего победителя.
"""
    ),
    code_cell(
        """overall_best = best_by_model.iloc[0]
overall_best_details = all_details[
    (all_details["Model"] == overall_best["Model"])
    & (all_details["Params"] == overall_best["Params"])
].sort_values(["EditDistance", "GT"], ascending=[False, True])

print("Лучший OCR:", overall_best["Model"])
print("Параметры:", overall_best["Params"])
display(
    overall_best_details[
        ["ImageKey", "Row", "GT", "Prediction", "Exact", "EditDistance", "CER", "RowFound"]
    ].head(25)
)
"""
    ),
    markdown_cell(
        """# Recognition-only benchmark на GT-вырезках

Предыдущий раздел честно измеряет весь существующий пайплайн, но его результат
сильно зависит от того, была ли строка вообще найдена.

Ниже разметка используется для вырезания области строки и определения её
направления. Вертикальные строки автоматически поворачиваются в горизонтальное
положение. Это **не production-пайплайн**, а отдельный тест, показывающий
потенциальное качество каждого OCR при правильной детекции строки.
"""
    ),
    code_cell(
        """GT_CROP_DIR = RESULT_DIR / "gt_crops"
GT_CROP_DIR.mkdir(exist_ok=True)


def rotate_bound(image, angle, background=0):
    image_h, image_w = image.shape
    center = (image_w / 2, image_h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_w = int(image_h * sine + image_w * cosine)
    new_h = int(image_h * cosine + image_w * sine)
    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderValue=background,
    )


def trim_foreground(image, padding=10):
    points = cv2.findNonZero((image > 0).astype(np.uint8))
    if points is None:
        return image
    x, y, w, h = cv2.boundingRect(points)
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(image.shape[1], x + w + padding), min(image.shape[0], y + h + padding)
    return image[y1:y2, x1:x2]


def gt_row_angle(row):
    if len(row["digits"]) < 2:
        return 0.0
    first = row["digits"][0]["center"]
    last = row["digits"][-1]["center"]
    return float(np.degrees(np.arctan2(last[1] - first[1], last[0] - first[0])))


def gt_row_crop(cleaned, row):
    crop, _ = crop_box(cleaned, row["box"], margin_ratio=0.15, min_margin=12)
    return trim_foreground(rotate_bound(crop, gt_row_angle(row)), padding=15)


def gt_pic7seg_prediction(cleaned, row):
    angle = gt_row_angle(row)
    predictions = []
    for digit in row["digits"]:
        digit_crop, _ = crop_box(
            cleaned,
            digit["box"],
            margin_ratio=0.08,
            min_margin=4,
        )
        digit_crop = trim_foreground(rotate_bound(digit_crop, angle), padding=5)
        predictions.append(recognize_pic7seg_digit(digit_crop))
    return "".join(predictions)


gt_crop_records = []
gt_crop_images = {}
for key in tqdm(pair_keys, desc="GT row crops"):
    binary = cv2.imread(str(binary_by_key[key]), cv2.IMREAD_GRAYSCALE)
    cleaned = apply_pic_morphology(binary)
    for row in load_gt_rows(json_by_key[key]):
        if not row["valid_text"]:
            continue
        record_id = f"{key}__{row['label'].replace(' ', '_')}"
        crop = gt_row_crop(cleaned, row)
        gt_crop_images[record_id] = crop
        cv2.imwrite(str(GT_CROP_DIR / f"{record_id}.png"), crop)
        gt_crop_records.append(
            {
                "RecordID": record_id,
                "ImageKey": key,
                "Row": row["label"],
                "GT": row["text"],
                "ValidText": True,
                "RowFound": True,
                "Pic7SegGTCrop": gt_pic7seg_prediction(cleaned, row),
                "Angle": gt_row_angle(row),
            }
        )

gt_valid_rows = pd.DataFrame(gt_crop_records)
gt_valid_rows.to_csv(RESULT_DIR / "gt_crop_row_dataset.csv", index=False)
print(f"GT-вырезок строк: {len(gt_valid_rows)}")
print(f"Поворачиваемых строк: {(gt_valid_rows['Angle'].abs() > 20).sum()}")
display(gt_valid_rows.head(10))
"""
    ),
    code_cell(
        """sample_gt_rows = gt_valid_rows[
    gt_valid_rows["Angle"].abs() > 20
].head(4)
sample_gt_rows = pd.concat(
    [sample_gt_rows, gt_valid_rows[gt_valid_rows["Angle"].abs() <= 20].head(8)]
).head(12)

figure, axes = plt.subplots(3, 4, figsize=(16, 9))
for axis, (_, row) in zip(axes.ravel(), sample_gt_rows.iterrows()):
    axis.imshow(gt_crop_images[row["RecordID"]], cmap="gray", vmin=0, vmax=255)
    axis.set_title(
        f"{row['ImageKey']} | {row['Row']}\\n"
        f"GT={row['GT']} | angle={row['Angle']:.1f} | Pic={row['Pic7SegGTCrop']}",
        fontsize=8,
    )
    axis.axis("off")
for axis in axes.ravel()[len(sample_gt_rows):]:
    axis.axis("off")
plt.tight_layout()
plt.show()
"""
    ),
    markdown_cell(
        """## Tesseract на GT-вырезках

Используется та же сетка параметров, что и в end-to-end тесте.
"""
    ),
    code_cell(
        """gt_tesseract_path = RESULT_DIR / "gtcrop_tesseract_details.csv"
if REUSE_OCR_RESULTS and gt_tesseract_path.exists():
    gt_tesseract_details = refresh_cached_metrics(
        load_detail_csv(gt_tesseract_path),
        gt_valid_rows,
    )
    print("Использованы сохранённые результаты Tesseract на GT-вырезках")
else:
    gt_tesseract_rows = []
    for psm, scale, polarity in tqdm(
        list(product(TESSERACT_PSMS, TESSERACT_SCALES, POLARITIES)),
        desc="Tesseract GT-crop grid",
    ):
        params = f"psm={psm};scale={scale};polarity={polarity}"
        for _, row in gt_valid_rows.iterrows():
            prepared = prepare_row_image(gt_crop_images[row["RecordID"]], polarity, scale)
            prediction = pytesseract.image_to_string(
                prepared,
                config=f"--psm {psm} -c tessedit_char_whitelist=0123456789",
            )
            gt_tesseract_rows.append(
                make_detail_row("Tesseract-GTCrop", params, row, prediction)
            )
    gt_tesseract_details = pd.DataFrame(gt_tesseract_rows)

gt_tesseract_summary = summarize_details(gt_tesseract_details)
gt_tesseract_details.to_csv(gt_tesseract_path, index=False)
gt_tesseract_summary.to_csv(RESULT_DIR / "gtcrop_tesseract_summary.csv", index=False)
display(gt_tesseract_summary.head(10).round(4))
"""
    ),
    markdown_cell(
        """## ssocr на GT-вырезках"""
    ),
    code_cell(
        """gt_ssocr_path = RESULT_DIR / "gtcrop_ssocr_details.csv"
if REUSE_OCR_RESULTS and gt_ssocr_path.exists():
    gt_ssocr_details = refresh_cached_metrics(
        load_detail_csv(gt_ssocr_path),
        gt_valid_rows,
    )
    print("Использованы сохранённые результаты ssocr на GT-вырезках")
else:
    gt_ssocr_rows = []
    gt_ssocr_grid = list(
        product(
            SSOCR_THRESHOLDS,
            SSOCR_MIN_SEGMENTS,
            SSOCR_ONE_RATIOS,
            SSOCR_OPERATIONS,
        )
    )
    for threshold, min_segment, one_ratio, operation in tqdm(
        gt_ssocr_grid,
        desc="ssocr GT-crop grid",
    ):
        params = (
            f"threshold={threshold};min_segment={min_segment};"
            f"one_ratio={one_ratio};operation={operation}"
        )
        for _, row in gt_valid_rows.iterrows():
            prepared = prepare_row_image(
                gt_crop_images[row["RecordID"]],
                polarity="black_on_white",
                scale=2,
            )
            prediction = run_ssocr(
                prepared,
                threshold=threshold,
                min_segment=min_segment,
                one_ratio=one_ratio,
                operation=operation,
            )
            gt_ssocr_rows.append(
                make_detail_row("ssocr-GTCrop", params, row, prediction)
            )
    gt_ssocr_details = pd.DataFrame(gt_ssocr_rows)

gt_ssocr_summary = summarize_details(gt_ssocr_details)
gt_ssocr_details.to_csv(gt_ssocr_path, index=False)
gt_ssocr_summary.to_csv(RESULT_DIR / "gtcrop_ssocr_summary.csv", index=False)
display(gt_ssocr_summary.head(10).round(4))
"""
    ),
    markdown_cell(
        """## EasyOCR и Pic7Seg на GT-вырезках"""
    ),
    code_cell(
        """gt_easyocr_path = RESULT_DIR / "gtcrop_easyocr_details.csv"
if REUSE_OCR_RESULTS and gt_easyocr_path.exists():
    gt_easyocr_details = refresh_cached_metrics(
        load_detail_csv(gt_easyocr_path),
        gt_valid_rows,
    )
    print("Использованы сохранённые результаты EasyOCR на GT-вырезках")
else:
    gt_easyocr_rows = []
    for decoder, scale, polarity in tqdm(
        list(product(EASYOCR_DECODERS, EASYOCR_SCALES, POLARITIES)),
        desc="EasyOCR GT-crop grid",
    ):
        params = f"decoder={decoder};scale={scale};polarity={polarity}"
        for _, row in gt_valid_rows.iterrows():
            prepared = prepare_row_image(gt_crop_images[row["RecordID"]], polarity, scale)
            prediction = run_easyocr(prepared, decoder)
            gt_easyocr_rows.append(
                make_detail_row("EasyOCR-GTCrop", params, row, prediction)
            )
    gt_easyocr_details = pd.DataFrame(gt_easyocr_rows)

gt_pic_details = pd.DataFrame(
    [
        make_detail_row(
            "Pic7Seg-GTCrop",
            "GT digit boxes + pic.ipynb recognizer",
            row,
            row["Pic7SegGTCrop"],
        )
        for _, row in gt_valid_rows.iterrows()
    ]
)
gt_easyocr_summary = summarize_details(gt_easyocr_details)
gt_pic_summary = summarize_details(gt_pic_details)
gt_easyocr_details.to_csv(gt_easyocr_path, index=False)
gt_easyocr_summary.to_csv(RESULT_DIR / "gtcrop_easyocr_summary.csv", index=False)
display(gt_easyocr_summary.head(10).round(4))
display(gt_pic_summary.round(4))
"""
    ),
    markdown_cell(
        """# Итог recognition-only benchmark

Эта таблица отвечает именно на вопрос: какой OCR лучше распознаёт строку, если
она уже корректно найдена, вырезана и ориентирована.
"""
    ),
    code_cell(
        """gt_all_details = pd.concat(
    [gt_pic_details, gt_tesseract_details, gt_ssocr_details, gt_easyocr_details],
    ignore_index=True,
)
gt_all_summary = pd.concat(
    [gt_pic_summary, gt_tesseract_summary, gt_ssocr_summary, gt_easyocr_summary],
    ignore_index=True,
)
gt_best_by_model = (
    gt_all_summary.sort_values(
        ["Model", "ExactRowAccuracy", "TotalCER", "NonEmptyRate"],
        ascending=[True, False, True, False],
    )
    .groupby("Model", as_index=False)
    .head(1)
    .sort_values(["ExactRowAccuracy", "TotalCER"], ascending=[False, True])
    .reset_index(drop=True)
)

gt_all_details.to_csv(RESULT_DIR / "gtcrop_all_ocr_details.csv", index=False)
gt_all_summary.to_csv(RESULT_DIR / "gtcrop_all_ocr_summary.csv", index=False)
gt_best_by_model.to_csv(RESULT_DIR / "gtcrop_best_ocr_by_model.csv", index=False)
display(gt_best_by_model.round(4))

figure, axes = plt.subplots(1, 2, figsize=(18, 6))
plot_data = gt_best_by_model.sort_values("ExactRowAccuracy")
axes[0].barh(plot_data["Model"], plot_data["ExactRowAccuracy"], color="seagreen")
axes[0].set_xlim(0, 1)
axes[0].set_title("Recognition-only: точность целой строки")
axes[0].grid(axis="x", alpha=0.25)

plot_data = gt_best_by_model.sort_values("TotalCER", ascending=False)
axes[1].barh(plot_data["Model"], plot_data["TotalCER"], color="firebrick")
axes[1].set_title("Recognition-only: CER, меньше — лучше")
axes[1].grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()
"""
    ),
]

# GT is allowed only for result matching and metrics. Remove the optional
# GT-crop benchmark from the emitted notebook.
for cell_index, cell in enumerate(cells):
    if (
        cell.get("cell_type") == "markdown"
        and "# Recognition-only benchmark на GT-вырезках" in "".join(cell["source"])
    ):
        cells = cells[:cell_index]
        break


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (.venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT_PATH.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1),
    encoding="utf-8",
)
print(f"Создан: {OUTPUT_PATH}")
