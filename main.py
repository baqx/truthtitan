from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import os
from kivy.core.text import LabelBase
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, FadeTransition
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.menu import MDDropdownMenu
import requests
from kivy.clock import Clock
#from kivy.config import Config
#Config.set('graphics', 'position', 'custom')
#Config.set('graphics', 'left', -1700)
#Config.set('graphics', 'top', 100)
from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.filemanager import MDFileManager
from kivymd.toast import toast
from kivymd.uix.snackbar import Snackbar
from kivy.core.window import Window
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
from io import BytesIO
import shutil
import os


Window.size = (320, 600)

# --------------------------------------------------------------------------------------------------------
Builder.load_file('kvs/pages/splash.kv')
Builder.load_file('kvs/pages/intro.kv')
Builder.load_file('kvs/pages/home.kv')
Builder.load_file('kvs/pages/detect.kv')
Builder.load_file('kvs/pages/result.kv')

# --------------------------------------------------------------------
# widgets
# ----------------------------------------------------------------------------------------------------------


class WindowManager(ScreenManager):
    pass


class SplashScreen(MDScreen):
    pass



class HomeScreen(MDScreen):
    pass

class IntroScreen(MDScreen):
    pass

class DetectScreen(MDScreen):
    l_active = BooleanProperty()
class ResultScreen(MDScreen):
    product=StringProperty()
    imgpath=StringProperty()
    csim=StringProperty()
    ssim=StringProperty()
    msg=StringProperty()
    
# --------------------------------------------------------------------------------------------------------------------------------
def create_private_folder():
    private_folder = 'private_folder'
    if not os.path.exists(private_folder):
        os.makedirs(private_folder)
    return private_folder

class MainApp(MDApp):
    dialog = None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)
        self.manager_open = False
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager, select_path=self.select_path, background_color_toolbar="#008080",preview=True,ext=['.png','.jpg']

        )

    def build(self):
        #Window.borderless = True
        self.my_theme_color = '#008080' #Primary Color
        self.my_theme_color1 = '#FF6F61' #Secondary Color
        self.bgcolor ='#D3D3D3'
        self.textcolor='#333333'
        self.cardbg='#F2F3F5'
        self.accent='#9900FF'
        self.theme_cls.theme_style = 'Light'
        self.theme_cls.primary_palette = 'Teal'
        self.theme_cls.accent_palette = 'Teal'
        self.theme_cls.accent_hue = '400'
        self.title = "Truth Titan"
        #self.theme_cls.material_style = "M3"

        self.wm = WindowManager(transition=FadeTransition())
        screens = [
            SplashScreen(name='Splash'),
            HomeScreen(name='Home'),
            IntroScreen(name='Intro'),
            DetectScreen(name='Detect'),
            ResultScreen(name='Result'),
            


        ]

        for screen in screens:
            self.wm.add_widget(screen)

        return self.wm
    
    def change_screen(self, screen):
        '''Change screen using the window manager.'''
        self.wm.current = screen

    def on_start(self):

        Clock.schedule_once(self.gotohome, 10)

    def gotohome(self, dt):
        self.change_screen("Home")
    
    def show_select_box(self):
        url = "https://explora.top/truthtitan/getcat.php"
        response = requests.get(url)
        if response.status_code == 200:
            categories = response.json()
            items = [category['name'] for category in categories]
        
        menu_items = [{"text": item, "viewclass": "OneLineListItem", "on_release": lambda x=item: self.update_selected_item(x)} for item in items]

        menu = MDDropdownMenu(
            caller=self.wm.get_screen('Detect').ids.product_cat,
            items=menu_items,
            width_mult=4,
        )
        menu.open()

    
    def update_selected_item(self, selected_item):
        self.wm.get_screen('Detect').ids.product_cat.text = selected_item

    def show_select_box2(self):
        url = "http://explora.top/truthtitan/getproducts.php"
        cname=self.wm.get_screen('Detect').ids.product_cat.text
        if cname=="":
            cname="Drinks"
        data = {"pass":"test","cname":cname}  # Change the value of cid as needed

        response = requests.post(url, data=data)

        if response.status_code == 200:
            products = response.json()
            items = [product['name'] for product in products]
        
        menu_items = [{"text": item, "viewclass": "OneLineListItem", "on_release": lambda x=item: self.update_selected_item2(x)} for item in items]

        menu = MDDropdownMenu(
            caller=self.wm.get_screen('Detect').ids.product_name,
            items=menu_items,
            width_mult=4,
        )
        menu.open()

    
    def update_selected_item2(self, selected_item):
        self.wm.get_screen('Detect').ids.product_name.text = selected_item
  

    def load_image_from_url(self,image_url):
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        return img
        
    def resize_image(self,image_path, target_size):
        img = Image.open(image_path)
        img_resized = img.resize(target_size, Image.LANCZOS)
        return img_resized

    def load_and_preprocess_image(self,image_path):
        img = keras_image.load_img(image_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def calculate_resnet50_features(self,  image_path):
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        img_array = self.load_and_preprocess_image(image_path)
        features = model.predict(img_array)
        return features.flatten()

    def calculate_ssim( self,original_path, fake_path):
        original_image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        fake_image = cv2.imread(fake_path, cv2.IMREAD_GRAYSCALE)

        # Get dimensions of original image
        target_size = (original_image.shape[1], original_image.shape[0])

        # Resize fake image to match original image dimensions
        fake_image_resized = self.resize_image(fake_path, target_size)
        fake_image_array = keras_image.img_to_array(fake_image_resized)
        fake_image_array = np.expand_dims(fake_image_array, axis=0)
        fake_image_array = preprocess_input(fake_image_array)

        # Calculate structural similarity index (SSIM)
        ssim_value, _ = ssim(original_image, fake_image_array[0, :, :, 0], data_range=fake_image_array[0, :, :, 0].max() - fake_image_array[0, :, :, 0].min(), full=True)
        return ssim_value 
        
    
    

    def detectproduct(self):
        try:
            print("Detecting")
            productname=self.wm.get_screen('Detect').ids.product_name.text
            url = "http://explora.top/truthtitan/getproduct.php"
            data = {"pname":productname,"pass":"test"}  # Change the value of name as needed

            response = requests.post(url, data=data)

            if response.status_code == 200:
                product_info = response.json()[0]  # Assuming the response is a list with a single item
                
                product_id = product_info.get("id")
                category_id = product_info.get("cid")
                name = product_info.get("name")
                image = product_info.get("img")
                description = product_info.get("description")
                
            else:
                print(f"Error: {response.status_code}")

            try:
                # Create a private folder
                private_folder = create_private_folder()

                original_image_url = image

                # Load original image from URL.
                original_image = self.load_image_from_url(original_image_url)

                # Save the original image as PNG in the private folder
                original_image_path = os.path.join(private_folder, 'original_image.png')
                original_image.save(original_image_path)
                
                fake_image_path = r"{}".format(self.wm.get_screen('Detect').ids.upbtn.text)
                if original_image_path!="" and fake_image_path!="":
                    self.wm.get_screen('Detect').l_active=True
                    # Calculate ResNet50 features
                    original_features = self.calculate_resnet50_features(original_image_path)
                    fake_features = self.calculate_resnet50_features(fake_image_path)

                    # Calculate cosine similarity of ResNet50 features
                    cosine_similarity = np.dot(original_features, fake_features) / (np.linalg.norm(original_features) * np.linalg.norm(fake_features))

                    # Calculate structural similarity index (SSIM)
                    ssim_value = self.calculate_ssim(original_image_path, fake_image_path)
                    # self.wm.get_screen('Detect').ids.loader.active="False"
                    print(f"Cosine Similarity: {cosine_similarity:.4f}")
                    print(f"SSIM: {ssim_value:.4f}")
                    self.wm.get_screen('Detect').l_active=False
                    self.wm.get_screen('Result').csim=f"{cosine_similarity:.4f}"
                    self.wm.get_screen('Result').ssim=f"{ssim_value:.4f}"
                    self.wm.get_screen('Result').product=productname
                    if cosine_similarity>=0.9:
                        msg=f"This product appears to be original due to its high cosine and structural similarity."
                    elif cosine_similarity<0.9 and cosine_similarity>=7:
                        msg=f"I have doubt that this product is original, please check other features of the product to confirm its reliability."
                    elif cosine_similarity<7 and cosine_similarity>=5:
                        msg=f"This product may be fake."
                    elif ssim_value<5:
                        msg=f"This product has a high probability of being fake. Use it at your own risk."
                    self.wm.get_screen('Result').msg=msg          
                    self.change_screen("Result")
                else:
                    #toast("Hello World")
                    if not self.dialog:
                        self.dialog = MDDialog(
                            text="You have not filled all the forms",
                            radius=[20, 7, 20, 7],
                            
                        )
                    self.dialog.open()
                self.wm.get_screen('Detect').l_active=False 
            except Exception as e:
                print(f"Error: {e}")
            finally:
                # Delete the private folder and its contents
                shutil.rmtree(private_folder, ignore_errors=True)
        except Exception as e:
            self.wm.get_screen('Detect').l_active=False
            print(f"Error: {e}")
















    def file_manager_open(self):
        self.file_manager.show(os.path.expanduser("~")) # output manager to the screen
        self.manager_open = True
        
    def select_path(self, path: str):
        '''
            It will be called when you click on the file name
            or the catalog selection button.
            :param path: path to the selected directory or file;
        '''
        self.exit_manager()
        toast(path)
        if path!="":
            self.wm.get_screen('Detect').ids.upbtn.text=str(path)
            self.wm.get_screen('Result').imgpath=str(path)
    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''
        self.manager_open = False
        self.file_manager.close()
    def events(self, instance, keyboard, keycode, text, modifiers):
        '''Called when buttons are pressed on the mobile device.'''
        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()
        return True

if __name__ == "__main__":
    LabelBase.register(name="swins", fn_regular="assets/fonts/swins.ttf")
    LabelBase.register(name="akadora", fn_regular="assets/fonts/akadora.ttf")
    LabelBase.register(name="firabold", fn_regular="assets/fonts/firabold.ttf")
    LabelBase.register(name="aff", fn_regular="assets/fonts/aff.ttf")
    LabelBase.register(name="firabook", fn_regular="assets/fonts/FiraSans-Book.ttf")
    LabelBase.register(name="firaregular", fn_regular="assets/fonts/FiraSans-Regular.ttf")
    LabelBase.register(name="firaebold", fn_regular="assets/fonts/FiraSans-ExtraBold.ttf")
    MainApp().run()
