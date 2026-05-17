import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def get_min_bbox(image, morph_kernel_size=5):
    img_np = np.array(image)
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    all_x, all_y = [], []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        all_x.extend([x, x + w])
        all_y.extend([y, y + h])
    min_col, max_col = min(all_x), max(all_x)
    min_row, max_row = min(all_y), max(all_y)
    expand = 5
    min_col = max(0, min_col - expand)
    min_row = max(0, min_row - expand)
    max_col = min(img_np.shape[1], max_col + expand)
    max_row = min(img_np.shape[0], max_row + expand)
    return image.crop((min_col, min_row, max_col, max_row))

def preprocess_image(image, target_size=(30, 30)):
    bbox_img = get_min_bbox(image)
    w, h = bbox_img.size
    max_dim = max(w, h)
    square_img = Image.new("L", (max_dim, max_dim), 255)
    paste_x, paste_y = (max_dim - w) // 2, (max_dim - h) // 2
    square_img.paste(bbox_img, (paste_x, paste_y))
    return square_img.resize(target_size, Image.LANCZOS)