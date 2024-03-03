from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import requests
from io import BytesIO
import shutil
import os

def load_image_from_url(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

def resize_image(image_path, target_size):
    img = Image.open(image_path)
    img_resized = img.resize(target_size, Image.LANCZOS)
    return img_resized

def load_and_preprocess_image(image_path):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def calculate_resnet50_features(image_path):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    img_array = load_and_preprocess_image(image_path)
    features = model.predict(img_array)
    return features.flatten()

def calculate_ssim(original_path, fake_path):
    original_image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    fake_image = cv2.imread(fake_path, cv2.IMREAD_GRAYSCALE)

    # Get dimensions of original image
    target_size = (original_image.shape[1], original_image.shape[0])

    # Resize fake image to match original image dimensions
    fake_image_resized = resize_image(fake_path, target_size)
    fake_image_array = keras_image.img_to_array(fake_image_resized)
    fake_image_array = np.expand_dims(fake_image_array, axis=0)
    fake_image_array = preprocess_input(fake_image_array)

    # Calculate structural similarity index (SSIM)
    ssim_value, _ = ssim(original_image, fake_image_array[0, :, :, 0], data_range=fake_image_array[0, :, :, 0].max() - fake_image_array[0, :, :, 0].min(), full=True)
    return ssim_value

def create_private_folder():
    private_folder = 'private_folder'
    if not os.path.exists(private_folder):
        os.makedirs(private_folder)
    return private_folder


def main():
    try:
        # Create a private folder
        private_folder = create_private_folder()

        original_image_url = 'http://explora.top/assets/img/avatars/robotss.png'

        # Load original image from URL
        original_image = load_image_from_url(original_image_url)

        # Save the original image as PNG in the private folder
        original_image_path = os.path.join(private_folder, 'original_image.png')
        original_image.save(original_image_path)

        # Fake image path with forward slashes
        fake_image_path = 'assets/img/logo.png'

        # Calculate ResNet50 features
        original_features = calculate_resnet50_features(original_image_path)
        fake_features = calculate_resnet50_features(fake_image_path)

        # Calculate cosine similarity of ResNet50 features
        cosine_similarity = np.dot(original_features, fake_features) / (np.linalg.norm(original_features) * np.linalg.norm(fake_features))

        # Calculate structural similarity index (SSIM)
        ssim_value = calculate_ssim(original_image_path, fake_image_path)

        print(f"Cosine Similarity: {cosine_similarity:.4f}")
        print(f"SSIM: {ssim_value:.4f}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Delete the private folder and its contents
        shutil.rmtree(private_folder, ignore_errors=True)

if __name__ == "__main__":
    main()
