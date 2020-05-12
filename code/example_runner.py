import numpy as np
from PIL import Image
from main_sampler import main_sampler

generated_images = main_sampler(['A man standing in a field','A girl standing in the ocean'],0)

for i in len(generated_images):
    im = Image.fromarray(generated_images[i])
    fullpath = './'+str(i)+'_.png'
    im.save(fullpath)

