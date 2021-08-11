import cv2
from PIL import Image

def ndarray2pil(img_list):
  img_pil_list = []
  for img in img_list:
    img_pil = Image.fromarray(img)
    img_pil_list.append(img_pil)
  return img_pil_list
