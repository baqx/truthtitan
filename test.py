from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivymd.app import MDApp
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import requests
from io import BytesIO

# Load KivyMD design from kv file
KV = '''
BoxLayout:
    orientation: 'vertical'



    BoxLayout:
        orientation: 'vertical'
        padding: dp(10)

        MDTextField:
            id: original_url_input
            hint_text: "Enter original image URL"
            size_hint_y: None
            height: dp(40)
            on_text_validate: app.calculate_similarity()

        MDTextField:
            id: fake_path_input
            hint_text: "Enter fake image path"
            size_hint_y: None
            height: dp(40)
            on_text_validate: app.calculate_similarity()

        MDLabel:
            id: similarity_result_label
            text: "Similarity: "
            theme_text_color: "Secondary"

        MDRaisedButton:
            text: "Calculate Similarity"
            on_release: app.calculate_similarity()
'''

class ImageSimilarityApp(MDApp):
    def build(self):
        return Builder.load_string(KV)

    def calculate_similarity(self):
        try:
            # Original image from URL
            original_image_url = self.root.ids.original_url_input.text

            # Load original image from URL
            original_image = self.load_and_preprocess_image_from_url(original_image_url)

            # Fake image from local path
            fake_image_path = self.root.ids.fake_path_input.text

            # Calculate ResNet50 features
            original_features = self.calculate_resnet50_features(original_image)
            fake_features = self.calculate_resnet50_features(fake_image_path)

            # Calculate cosine similarity of ResNet50 features
            cosine_similarity = np.dot(original_features, fake_features) / (
                    np.linalg.norm(original_features) * np.linalg.norm(fake_features))

            # Calculate structural similarity index (SSIM)
            ssim_value = self.calculate_ssim_from_url(original_image_url, fake_image_path)

            # Display results
            similarity_result_label = self.root.ids.similarity_result_label
            similarity_result_label.text = f"Similarity: {cosine_similarity:.4f} | SSIM: {ssim_value:.4f}"

        except Exception as e:
            print(f"Error: {e}")

    def load_and_preprocess_image_from_url(self, image_url):
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224))  # Resize to match ResNet50 input size
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def calculate_resnet50_features(self, img_array):
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        features = model.predict(img_array)
        return features.flatten()

    def calculate_ssim_from_url(self, original_url, fake_path):
        original_image_array = self.load_and_preprocess_image_from_url(original_url)
        fake_image = Image.open(fake_path)
        fake_image_array = keras_image.img_to_array(fake_image)
        fake_image_array = np.expand_dims(fake_image_array, axis=0)
        fake_image_array = preprocess_input(fake_image_array)

        # Calculate structural similarity index (SSIM)
        ssim_value, _ = ssim(original_image_array[0, :, :, 0], fake_image_array[0, :, :, 0],
                             data_range=fake_image_array[0, :, :, 0].max() - fake_image_array[0, :, :, 0].min(),
                             full=True)
        return ssim_value


if __name__ == "__main__":
    ImageSimilarityApp().run()
