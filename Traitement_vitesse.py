import os
from pathlib import Path
import cv2
import numpy as np


INPUT_PATH = "Vidéos_traitées"
OUTPUT_ROOT = "video_traitee_vitesse"

REAL_FPS = 30
PIXEL_SIZE = 4.43e-4
OUTPUT_VIDEO_FPS = 4

RESIZE_FACTOR = 0.7

PYR_SCALE = 0.5
LEVELS = 3
WINSIZE = 15
ITERATIONS = 3
POLY_N = 5
POLY_SIGMA = 1.2

CONTOUR_METHOD = "background_gradient"
CONTOUR_MIN_AREA_RATIO = 0.002


def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)


def is_video_file(p):
    return p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]


def preprocess(gray):
    clahe = cv2.createCLAHE(2.0, (8, 8))
    gray = clahe.apply(gray)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def detect_circle(gray):
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=int(min(gray.shape) * 0.3),
        maxRadius=int(min(gray.shape) * 0.6),
    )

    if circles is None:
        raise RuntimeError("Cercle non détecté")

    return np.uint16(np.around(circles[0][0]))


def build_mask(shape, cx, cy, r):
    mask = np.zeros(shape, np.uint8)
    cv2.circle(mask, (cx, cy), r + 10, 255, -1)
    return mask


def crop(gray, cx, cy, r):
    x1 = max(cx - r, 0)
    x2 = min(cx + r, gray.shape[1])
    y1 = max(cy - r, 0)
    y2 = min(cy + r, gray.shape[0])
    return gray[y1:y2, x1:x2]


def speed_to_colormap(vx, vy, mask):
    speed = np.sqrt(vx**2 + vy**2)
    speed = cv2.GaussianBlur(speed, (7, 7), 0)

    valid = speed[mask > 0]

    if valid.size == 0:
        valid = speed.ravel()

    threshold = np.percentile(valid, 70)
    speed[speed < threshold] = 0

    vmax = np.percentile(valid, 98)
    if vmax <= 0:
        vmax = 1

    norm = np.clip(speed / vmax, 0, 1)
    img = (norm * 255).astype(np.uint8)

    color = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    color[mask == 0] = 0

    return color, speed


def get_contour(gray, background, speed, mask):
    diff = cv2.absdiff(gray, background)
    diff = cv2.GaussianBlur(diff, (7, 7), 0)

    gradx = cv2.Sobel(diff, cv2.CV_32F, 1, 0)
    grady = cv2.Sobel(diff, cv2.CV_32F, 0, 1)
    grad = cv2.magnitude(gradx, grady)

    # normalisation
    valid = grad[mask > 0]
    t = np.percentile(valid, 85)

    _, binary = cv2.threshold(grad.astype(np.uint8), t, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    min_area = CONTOUR_MIN_AREA_RATIO * gray.shape[0] * gray.shape[1]

    best = None
    best_area = 0

    for c in contours:
        a = cv2.contourArea(c)
        if a > best_area and a > min_area:
            best = c
            best_area = a

    return best


def estimate_background(video, cx, cy, r):
    cap = cv2.VideoCapture(str(video))
    samples = []

    for i in range(30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * 10)
        ret, f = cap.read()
        if not ret:
            continue

        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

        samples.append(crop(g, cx, cy, r))

    cap.release()
    return np.median(np.stack(samples), axis=0).astype(np.uint8)


def process(video):
    video = Path(video)
    name = video.stem

    out_dir = Path(OUTPUT_ROOT) / name
    safe_mkdir(out_dir)

    cap = cv2.VideoCapture(str(video))

    ret, f0 = cap.read()
    if not ret:
        return

    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    g0 = cv2.resize(g0, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

    cx, cy, r = detect_circle(g0)

    mask = build_mask(g0.shape, cx, cy, r)

    background = estimate_background(video, cx, cy, r)

    prev = preprocess(crop(g0, cx, cy, r))

    h, w = prev.shape

    writer = cv2.VideoWriter(
        str(out_dir / "colormap_contour_4fps.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_VIDEO_FPS,
        (w, h),
    )

    dt = 1 / REAL_FPS

    while True:
        ret, f = cap.read()
        if not ret:
            break

        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

        g_crop = crop(g, cx, cy, r)
        g_proc = preprocess(g_crop)

        flow = cv2.calcOpticalFlowFarneback(
            prev, g_proc, None,
            PYR_SCALE, LEVELS, WINSIZE,
            ITERATIONS, POLY_N, POLY_SIGMA, 0
        )

        vx = flow[..., 0] / dt * PIXEL_SIZE
        vy = flow[..., 1] / dt * PIXEL_SIZE

        color, speed = speed_to_colormap(vx, vy, crop(mask, cx, cy, r))

        contour = get_contour(g_crop, background, speed, crop(mask, cx, cy, r))

        if contour is not None:
            cv2.drawContours(color, [contour], -1, (255, 255, 255), 2)

        writer.write(color)

        prev = g_proc

    cap.release()
    writer.release()

    print("terminé:", name)


def main():
    p = Path(INPUT_PATH)
    safe_mkdir(OUTPUT_ROOT)

    if p.is_file():
        process(p)

    else:
        for v in p.iterdir():
            if is_video_file(v):
                process(v)


if __name__ == "__main__":
    main()