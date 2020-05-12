import numpy as np
from PIL import Image
from main_sampler import main_sampler

gpu_id = 0

sentences = ['A man standing in a field',
             'A girl standing in the ocean']

generated_images = main_sampler(sentences, gpu_id)

for i in range(len(generated_images)):
    im = Image.fromarray(generated_images[i])
    fullpath = './'+str(i)+'_.png'
    im.save(fullpath)

