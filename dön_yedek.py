import random

from pathlib import Path

import numpy as np

from hailo_sdk_client import ClientRunner

from PIL import Image

from ultralytics import YOLO

from ultralytics.data.utils import check_det_dataset

from ultralytics.utils import DATASETS_DIR


# Configuration

MODEL = "can_yoloV8m"

HW_ARCH = "hailo8"  # hailo8 | hailo8l | hailo15h

IMGSZ = 640

CALIB_IMAGES = 256

NMS_CONFIG = "hailo_yolov8m_480_640_nms_config.json" 


# YOLO detection head end nodes. 

END_NODES = [

    "/model.22/cv2.0/cv2.0.2/Conv",

    "/model.22/cv3.0/cv3.0.2/Conv",

    "/model.22/cv2.1/cv2.1.2/Conv",

    "/model.22/cv3.1/cv3.1.2/Conv",

    "/model.22/cv2.2/cv2.2.2/Conv",

    "/model.22/cv3.2/cv3.2.2/Conv",

]


# Step 1: Export to ONNX

model = YOLO(f"{MODEL}.pt")

model.export(format="onnx", imgsz=IMGSZ, opset=11)


# Step 2: Parse ONNX with Hailo DFC

runner = ClientRunner(hw_arch=HW_ARCH)

runner.translate_onnx_model(f"{MODEL}.onnx", end_node_names=END_NODES)


# Step 3: Load model script (normalization + HailoRT NMS)

# DİKKAT: performance_param komutu maksimum donanım optimizasyonu için eklendi.

alls = (

    "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"

    "change_output_activation(conv54, sigmoid)\n"

    "change_output_activation(conv65, sigmoid)\n"

    "change_output_activation(conv80, sigmoid)\n"

    f'nms_postprocess("{NMS_CONFIG}", meta_arch=yolov8, engine=cpu)\n'

    "allocator_param(width_splitter_defuse=disabled)\n"

    "performance_param(compiler_optimization_level=max)\n"

)

runner.load_model_script(alls)


# Step 4: Build calibration dataset

check_det_dataset("canM.yaml")

calib_dir = Path("/home/ahmet/Desktop/AİKİT/can_yoloV8M/kalibrasyon")

image_files = list(calib_dir.glob("*.jpg")) + list(calib_dir.glob("*.png"))


if not image_files:

    raise FileNotFoundError(f"No calibration images found in {calib_dir}")


# DÜZELTME: Mükerrer resim seçimini önlemek için listeyi karıştırıp ilk 256'sını alıyoruz.

random.shuffle(image_files)

actual_calib_count = min(CALIB_IMAGES, len(image_files))


print(f"\n[BİLGİ] Kalibrasyon için {actual_calib_count} adet BENZERSİZ görsel yükleniyor...\n")


calibset = np.zeros((actual_calib_count, IMGSZ, IMGSZ, 3), dtype=np.float32)

for i in range(actual_calib_count):

    img = Image.open(image_files[i]).convert("RGB").resize((IMGSZ, IMGSZ))

    calibset[i] = np.array(img, dtype=np.float32)


# Step 5: Optimize and quantize

runner.optimize(calibset)

runner.save_har(f"{MODEL}.o.har")


# Step 6: Compile to HEF

hef = runner.compile()

with open(f"{MODEL}.hef", "wb") as f:

    f.write(hef)


print(f"Compiled HEF saved to: {MODEL}.hef")
