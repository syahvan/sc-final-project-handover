import base64
import numpy as np
from model import Model
from preprocessing import Preprocessing
from inference import Inference


class Main:
    def __init__(self, img_input):
        self.img_input = img_input
        self.inference()
        self.postpreprocessing()

    def get_model(self):
        model = Model()
        model = model.get_model
        return model

    def prepare_input(self):
        prep = Preprocessing(img_name=self.img_input)
        prep.load_image()
        prep = prep.get_image_
        return prep

    def inference(self):
        model = self.get_model()
        prep = self.prepare_input()
        print(model)
        model = Inference(model, prep)
        self.result = model.infer()

    def postpreprocessing(self):
        dict_map = {
            0:"Angle Boot",
            1:"Bag",
            2:"Coat",
            3:"Dress",
            4:"Pullover",
            5:"Sandals",
            6:"Shirt",
            7:"Sneaker",
            8:"Trouser",
            9:"T-shirt",
            10:"Hat"
        }
        label = dict_map[self.result.tolist()[0]]
        self._label = label

    @property
    def get_results(self):
        return self._label



if __name__ == "__main__":
    print("Welcome to classifier\nplease input the model and image path")
    img_path = input("Image Path : ")


    binary_fc       = open(img_path, 'rb').read()
    base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')
    print(f"{img_path}, base64: {True}")

    model_img = Main(img_input=base64_utf8_str)
    print(model_img.get_results)
