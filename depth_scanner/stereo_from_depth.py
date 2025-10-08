import os

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()
PATH_FOLDER_INPUT = os.getenv('PATH_FOLDER_INPUT')
PATH_FOLDER_OUTPUT = os.getenv('PATH_FOLDER_OUTPUT')
PATH_FOLDER_STEREO = os.getenv('PATH_FOLDER_STEREO')

# Rutas
frames_dir = PATH_FOLDER_INPUT
depth_dir = PATH_FOLDER_OUTPUT
stereo_out_dir = PATH_FOLDER_STEREO
os.makedirs(stereo_out_dir, exist_ok=True)

# Parámetro configurable: intensidad del efecto 3D
disparity_strength = 0.06  # Ajustalo entre 0.02 y 0.1 para más o menos profundidad

# Procesar cada frame + depth
image_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
print(f"Found {len(image_files)} images to process")

for idx, fname in enumerate(image_files, 1):
    print(f"Processing {idx}/{len(image_files)}: {fname}")

    # Leer imagen y mapa de profundidad
    img = cv2.imread(os.path.join(frames_dir, fname))
    depth = cv2.imread(os.path.join(depth_dir, fname), cv2.IMREAD_GRAYSCALE)
    
    # Resize depth map to match image dimensions if needed
    if depth.shape != img.shape[:2]:
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Normalizar profundidad a 0-1
    depth_norm = depth.astype(np.float32) / 255.0

    # Generar mapa de desplazamiento
    shift = (depth_norm - 0.5) * 2 * disparity_strength * img.shape[1]  # desplazamiento en píxeles

    # Crear imágenes izquierda y derecha
    print(f"Creating left and right images")
    left_img = np.zeros_like(img)
    right_img = np.zeros_like(img)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            dx = int(shift[y, x])
            # Izquierda: mové hacia la derecha (simula ojo izquierdo)
            if 0 <= x + dx < img.shape[1]:
                left_img[y, x] = img[y, x + dx]
            # Derecha: mové hacia la izquierda (simula ojo derecho)
            if 0 <= x - dx < img.shape[1]:
                right_img[y, x] = img[y, x - dx]

    # Combinar side-by-side
    output_path = os.path.join(stereo_out_dir, fname)
    print(f"Creating stereo + save: '{output_path}'")
    stereo = np.hstack([left_img, right_img])
    try:
        cv2.imwrite(output_path, stereo)
    except Exception as e:
        print(f"error: {e}")
