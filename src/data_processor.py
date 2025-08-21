import cv2
from ultralytics import YOLO
import csv
import os
import numpy as np
import yaml
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_PATH = "C:/Users/S.Arjun/Downloads/student behavior.v15i.yolov8"
OUTPUT_CSV_FILE = os.path.join('data', 'classroom_actions.csv')
os.makedirs('data', exist_ok=True)

# --- LABEL MAPPING ---
LABEL_MAPPING = {
    'sit': 'ATTENTIVE',
    'lookup': 'ATTENTIVE',
    'bow': 'INATTENTIVE',
    'down': 'SLEEPING'
}

# --- SCRIPT ---
model = YOLO('yolov8n-pose.pt')

try:
    with open(os.path.join(DATASET_PATH, 'data.yaml'), 'r') as f:
        data_yaml = yaml.safe_load(f)
    class_names = data_yaml['names']
    print("--- Found Labels in data.yaml ---")
    print(class_names)
    print("---------------------------------")
except Exception as e:
    print(f"ERROR: Could not read data.yaml. Please check the DATASET_PATH: {e}")
    exit()

header = ['label'] + [f'kp_{i}_{v}' for i in range(33) for v in ['x', 'y', 'conf']]
with open(OUTPUT_CSV_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

rows_written = 0

for split in ['train', 'valid', 'test']:
    image_folder = os.path.join(DATASET_PATH, split, 'images')
    label_folder = os.path.join(DATASET_PATH, split, 'labels')

    if not os.path.isdir(image_folder):
        print(f"Info: '{split}' folder not found. Skipping.")
        continue

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\nProcessing {len(image_files)} images in '{split}' split...")

    for image_file in tqdm(image_files, desc=f"Processing {split}"):
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, os.path.splitext(image_file)[0] + '.txt')

        frame = cv2.imread(image_path)
        if frame is None:
            continue

        results = model(frame, verbose=False)

        if os.path.isfile(label_path) and results[0].keypoints.data.numel() > 0:
            try:
                with open(label_path, 'r') as f:
                    first_line = f.readline()
                    if not first_line: continue
                    class_id = int(first_line.split()[0])
                    original_label = class_names[class_id]

                if original_label in LABEL_MAPPING:
                    our_label = LABEL_MAPPING[original_label]
                    keypoints = results[0].keypoints.data[0].cpu().numpy()
                    keypoints_row = keypoints.flatten()

                    # --- FIX: Combine label and data in a standard Python list ---
                    labeled_row = [our_label] + keypoints_row.tolist()

                    with open(OUTPUT_CSV_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(labeled_row)
                    rows_written += 1
            except Exception:
                continue

print(f"\n--- PROCESSING SUMMARY ---")
if rows_written > 0:
    print(f"✅ Success! Wrote {rows_written} rows to the dataset.")
    print(f"Data saved to {OUTPUT_CSV_FILE}")
else:
    print(f"❌ Warning: No matching labels were found.")
    print("Please check the 'Found Labels' printed above against the LABEL_MAPPING in the script.")
print("--------------------------")
