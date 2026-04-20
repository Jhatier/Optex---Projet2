import cv2
import numpy as np
from pathlib import Path

# ========= PARAMÈTRES =========
input_folder = Path("videos_entree")
output_folder = Path("videos_traitees")
output_folder.mkdir(exist_ok=True)

video_extensions = {".mp4", ".mov", ".avi", ".mkv"}

# seuil pour détecter le fond noir
background_threshold = 25

# renforcement de contraste dans la zone utile
alpha = 1.4   # contraste
beta = -10    # luminosité

# ========= FONCTION DE TRAITEMENT =========
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection du fond noir
    background_mask = gray < background_threshold

    # Amélioration douce du contraste
    enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Image finale :
    # - fond -> blanc
    # - reste -> niveaux de gris normaux
    result = enhanced.copy()
    result[background_mask] = 255

    return result

# ========= TRAITEMENT DES VIDÉOS =========
for video_path in input_folder.iterdir():
    if video_path.suffix.lower() not in video_extensions:
        continue

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Impossible d'ouvrir {video_path.name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = output_folder / f"{video_path.stem}_processed.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
        isColor=False
    )

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = process_frame(frame)
        writer.write(processed)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"{video_path.name} traité : {frame_count} frames")

print("Terminé.")