import cv2
import numpy as np
from pathlib import Path

#Ce fichier permet de cropper les vidéos en carré, histoire de pouvoir avoir un bon format de vidéo

video_path = Path("Vidéos_brutes/Bougie-rose (2).MOV")

output_folder = Path("Vidéos_traitées")
output_folder.mkdir(exist_ok=True)

display_scale = 0.4
background_threshold = 25 #Paramètres de contraste pour différencier l'image du miroir et permettre de cadrer l'image
show_preview = True 

clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)) #Format de contraste


video = cv2.VideoCapture(str(video_path))
if not video.isOpened():
    print(f"Impossible d'ouvrir {video_path}")
        
fps = video.get(cv2.CAP_PROP_FPS)

ret, frame = video.read()
if not ret:
    print(f"Aucune frame lisible dans {video_path.name}")
    video.release()
    
h, w = frame.shape[:2] #Centrer l'image 
size = min(h, w)
center_y, center_x = h // 2, w // 2
y1 = center_y - size // 2
y2 = center_y + size // 2
x1 = center_x - size // 2
x2 = center_x + size // 2

frame_cropped = frame[y1:y2, x1:x2]
crop_h, crop_w = frame_cropped.shape[:2]

output_path = output_folder / f"{video_path.stem}_traitee.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(output_path), fourcc, fps, (crop_w, crop_h), isColor=False)

video.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    size = min(h, w)
    center_y, center_x = h // 2, w // 2
    y1 = center_y - size // 2
    y2 = center_y + size // 2
    x1 = center_x - size // 2
    x2 = center_x + size // 2

    frame_cropped = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2GRAY)

    enhanced = clahe.apply(gray)

    background_mask = gray < background_threshold
    result = enhanced.copy()
    result[background_mask] = 255

    writer.write(result)

    if show_preview:
        frame_resized = cv2.resize(result, None, fx=display_scale, fy=display_scale)
        cv2.imshow("Frame traitée", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
writer.release()
print(f"Terminé : {output_path.name}")

cv2.destroyAllWindows()
print("Toutes les vidéos ont été traitées.")