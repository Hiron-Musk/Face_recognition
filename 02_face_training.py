import cv2
import numpy as np
from PIL import Image #오류시 pip install pillow로 라이브러리 설치하기
import os


path = 'C:/data/user_info/'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier()
detector.load('C:\Cascades\haarcascade_frontalface_default.xml')#경로 수정


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') 
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write('C:/data/trainingface/trainer.yml') #yml파일 저장경로
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))