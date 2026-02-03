import os
import cv2
import numpy as np
import albumentations as A

# =========================
# CONFIGURATION
# =========================
BASE_DIR = "dataset"
CLASSES = ["healthy", "unhealthy"]
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

# =========================
# AUGMENTATION PIPELINE
# =========================
augmentations = {
    "brightness": A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0, p=1),
    "contrast": A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.3, p=1),
    "crop": A.RandomResizedCrop(size=(224, 224), scale=(0.7, 0.9), p=1),
    "flip_h": A.HorizontalFlip(p=1),
    "flip_v": A.VerticalFlip(p=1),
    "gaussian_noise": A.GaussNoise(std_range=(0.1, 0.5), p=1),
    "poisson_noise": A.ISONoise(p=1),
    "rotation_15": A.Rotate(limit=15, p=1),
    "saturation": A.HueSaturationValue(sat_shift_limit=30, p=1),
    "jitter": A.ColorJitter(p=1),
    "hist_equalized": A.CLAHE(p=1),
    "translated": A.Affine(translate_percent=0.1, p=1),
    "unsharp": A.Sharpen(alpha=(0.2, 0.5), p=1),
}

# =========================
# CUSTOM FILTERS
# =========================
def sobel_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel = cv2.magnitude(sx, sy)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

def laplacian_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.uint8(np.clip(np.abs(lap), 0, 255))
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

def high_pass_filter(img):
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

# =========================
# MAIN PROCESS
# =========================
def augment_class(class_name):
    input_dir = os.path.join(BASE_DIR, class_name, "merged")
    output_dir = os.path.join(BASE_DIR, class_name, "augmented")
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if not file.lower().endswith(IMG_EXTENSIONS):
            continue

        img_path = os.path.join(input_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        name, ext = os.path.splitext(file)

        # Albumentations
        for aug_name, aug in augmentations.items():
            augmented = aug(image=img)["image"]
            out_name = f"{name}__{aug_name}{ext}"
            cv2.imwrite(os.path.join(output_dir, out_name), augmented)

        # Custom Filters
        cv2.imwrite(os.path.join(output_dir, f"{name}__sobel{ext}"), sobel_filter(img))
        cv2.imwrite(os.path.join(output_dir, f"{name}__laplacian{ext}"), laplacian_filter(img))
        cv2.imwrite(os.path.join(output_dir, f"{name}__highpass{ext}"), high_pass_filter(img))


def main():
    for cls in CLASSES:
        print(f"Processing class: {cls}")
        augment_class(cls)
    print("✅ Augmentation complete.")

if __name__ == "__main__":
    main()
