from django.http import HttpResponse
from django.shortcuts import render
from django.db import models
import pickle
# Create your views here.
import  os


from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from PIL import Image as Img
import numpy as np
import random as rnd
def main(request):
    cls="__"
    file = "static/img/img.png"
    labels = ['Daffodil', 'Daisy', 'Iris', 'Sunflower', 'Windflower']
    if request.method == 'POST':

        if 'file' not in request.FILES:
            print("No Image")
            return render(request,'web/hi.html',{"title":"Flowers Classification","class":cls,"src":file})
        else:
            file = request.FILES['file']

        if file == None:
            return;
        #data = file.read()
        image = Img.open(file)

        img_arr = np.asarray(image)
        #print(image)
        rndfilename =   '/static/temp/temp_'+str(rnd.randint(1,99999999999))+'.png'
        CURRENT_DIR = os.path.dirname(__file__)
        model_file = os.path.join(CURRENT_DIR, 'model/finalized_model.sav')
        image.save(CURRENT_DIR+rndfilename,"PNG")
        file = rndfilename
        loaded_model = pickle.load(open(model_file, 'rb'))

        labels = ['Daffodil', 'Daisy', 'Iris', 'Sunflower', 'Windflower']
        img_size=224
        #img_arr = imread(os.path.join(CURRENT_DIR, 'model/image_0726.jpg'))
        resized_arr = resize(img_arr, (img_size, img_size))
        fd, hog_image = hog(resized_arr, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,multichannel=True)

        result = loaded_model.predict([fd])
        cls=str(labels[result[0]])


    return render(request,'web/hi.html',{"title":"Flowers Classification","class":cls,"src":file})