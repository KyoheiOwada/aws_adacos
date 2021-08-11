import cv2
import numpy as np
import glob
from PIL import Image

def detect_face(img):
  face_cascade = cv2.CascadeClassifier('/face_recognition/utils/haarcascade_frontalface_default.xml')
  face_img = img.copy()
  img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
  print(face_img.shape)
  face_rects = face_cascade.detectMultiScale(img_gray, scaleFactor=1.25, minNeighbors=18)
  img_array_list =[]
  if face_rects == []:
    return img_array_list
  for i, (x,y,w,h) in enumerate(face_rects):
    print(x,y,w,h)
    if x >= face_img.shape[0] or y >= face_img.shape[1]:
      continue
    rect_face_img = face_img[y:y+h, x:x+w]
    rect_face_img = cv2.cvtColor(rect_face_img, cv2.COLOR_BGR2RGB)
    rect_face_img_array = np.array(rect_face_img)
    print(rect_face_img.shape)
    #rect_face_img_pil = Image.fromarray(rect_face_img_array)
    img_array_list.append(rect_face_img_array)
    file_name = 'face'+ str(i).zfill(8) + '.png'
    #cv2.imshow('test', rect_face_img_array)
    #cv2.waitKey(1000)
    #cv2.imwrite(file_name, rect_face_img_array)
    #cv2.imshow("image", rect_face_img)
  return img_array_list

if __name__ == '__main__':
  frame_path_list = glob.glob('./frame/*.jpg')
  for frame in frame_path_list:
    img = cv2.imread(frame)
    face_imgs = detect_face(img)
  for face in face_imgs:
    if face == None:
      break
    else:
      print('end')
      #cv2.imshow("image", face)
