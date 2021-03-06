# -*- coding: utf-8 -*-
"""Make_Augmentation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z-ed1mQOPYjTmIoEm0tI48CGuJwe2TxJ
"""

#tạo Augmentation
path0 = "/content/drive/MyDrive/DataAI_t/"
path1 = "/content/drive/MyDrive/DataAI_t/Compass/"
path2 = "/content/drive/MyDrive/DataAI_t/Human/"
path3 = "/content/drive/MyDrive/DataAI_t/Lion/"
path4 = "/content/drive/MyDrive/DataAI_t/Sun/"
path = path0

loadedImages = []
imagesList = listdir(path)
for image in imagesList:
  img = PImage.open(datapath + '/' + image)
  img = img.convert('RGB')
  # img.show()
  # print(image)
  datagen = ImageDataGenerator(rotation_range = 0,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True,brightness_range = (0.5, 1.5))
  x = img_to_array(img)
  x = x.reshape((1, ) + x.shape)
  i = 0
  for batch in datagen.flow(x, batch_size = 1,save_to_dir = datapath, save_prefix ='Lion'+ str(i) , save_format ='jpg'):
    i += 1
    if i > 5:
        break 
  loadedImages.append(img)