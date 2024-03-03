from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import requests
from io import BytesIO

def load_and_preprocess_image_from_url(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))  # Resize to match ResNet50 input size
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def calculate_resnet50_features(img_array):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    features = model.predict(img_array)
    return features.flatten()

def calculate_ssim_from_url(original_url, fake_path):
    original_image_array = load_and_preprocess_image_from_url(original_url)
    fake_image = Image.open(fake_path)
    fake_image_array = keras_image.img_to_array(fake_image)
    fake_image_array = np.expand_dims(fake_image_array, axis=0)
    fake_image_array = preprocess_input(fake_image_array)

    # Calculate structural similarity index (SSIM)
    ssim_value, _ = ssim(original_image_array[0, :, :, 0], fake_image_array[0, :, :, 0], data_range=fake_image_array[0, :, :, 0].max() - fake_image_array[0, :, :, 0].min(), full=True)
    return ssim_value

def main():
    try:
        # Original image from URL
        original_image_url = 'http://explora.top/assets/img/avatars/robotss.png'

        # Load original image from URL
        original_image = load_and_preprocess_image_from_url(original_image_url)

        # Fake image from local path
        fake_image_path = 'C:\Users\use\Downloads\cocktailier-main\cocktailier-main\assets\img\logo.png'

        # Calculate ResNet50 features
        original_features = calculate_resnet50_features(original_image)
        fake_features = calculate_resnet50_features(fake_image_path)

        # Calculate cosine similarity of ResNet50 features
        cosine_similarity = np.dot(original_features, fake_features) / (np.linalg.norm(original_features) * np.linalg.norm(fake_features))

        # Calculate structural similarity index (SSIM)
        ssim_value = calculate_ssim_from_url(original_image_url, fake_image_path)

        print(f"Cosine Similarity: {cosine_similarity:.4f}")
        print(f"SSIM: {ssim_value:.4f}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
