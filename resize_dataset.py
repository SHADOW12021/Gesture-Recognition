import os
import random
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

SOURCE_DIR = r"D:\HaGRIDv2_dataset_512\point"
DEST_DIR = r"C:\Users\gsocc\OneDrive\Desktop\Digital Image Processing\FINAL\HandGrid approach\hagrid-classification-512p\point"

NUM_IMAGES = 28000
TARGET_SIZE = 512
NUM_THREADS = 8

os.makedirs(DEST_DIR, exist_ok=True)

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
all_images = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(valid_extensions)]

selected_images = random.sample(all_images, min(NUM_IMAGES, len(all_images)))

def resize_and_crop(img):
    width, height = img.size

    # scale so smallest side = 512
    scale = TARGET_SIZE / min(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    img = img.resize((new_width, new_height), Image.LANCZOS)

    # center crop
    left = (new_width - TARGET_SIZE) // 2
    top = (new_height - TARGET_SIZE) // 2
    right = left + TARGET_SIZE
    bottom = top + TARGET_SIZE

    img = img.crop((left, top, right, bottom))
    return img

def process_image(filename):
    try:
        src_path = os.path.join(SOURCE_DIR, filename)
        dest_path = os.path.join(DEST_DIR, filename)

        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = resize_and_crop(img)
            img.save(dest_path, "JPEG", quality=95)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    executor.map(process_image, selected_images)

print("✅ Done! Images resized without distortion.")