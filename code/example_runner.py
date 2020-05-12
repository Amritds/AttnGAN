import numpy as np
from PIL import Image
from main_sampler import main_sampler

sentences = ['A man standing in a field',
             'A girl standing in the ocean']
print('\n\nGenerating Images now\n\n')
generated_images = main_sampler(sentences)

for i in range(len(generated_images)):
    im = Image.fromarray(generated_images[i])
    fullpath = './'+str(i)+'_.png'
    im.save(fullpath)

