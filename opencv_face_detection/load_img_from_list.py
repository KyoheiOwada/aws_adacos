import glob
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

dir_name = 'images'
file_type = 'jpg'

img_list = glob.glob(dir_name + '/*.' + file_type)
temp_img_array_list = []

for img in img_list:
  temp_img = Image.open(img)
  temp_image_array = np.array(temp_img, dtype='float32') / 255
  temp_img_array_list.append(temp_img_array)
