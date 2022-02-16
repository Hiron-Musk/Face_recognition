#얼굴인식된 사진은 저장되나 카메라가 켜지지 않는 오류 발생

import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier()
faceCascade.load('C:\Cascades\haarcascade_frontalface_default.xml')#경로변경

face_id = input('\n유저 ID를 입력해주세요->')

def face_extractor(img):
     
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray,1.3,5)
    
    if faces is():
        return None
     
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


cap = cv2.VideoCapture(0)

count = 0
while True:
     
    ret, frame = cap.read()
    
    if face_extractor(frame) is not None:
        count+=1
        
        face = cv2.resize(face_extractor(frame),(200,200))
        
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        #웹캠의 얼굴 캡쳐 저장경로
        file_name_path = 'C:/data/user_info/User.'+str(face_id)+'.'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)
        
             
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)

    else:
        print("Face not Found")
        pass

    if cv2.waitKey(1)==13 or count==15:
        break

cap.release()
cv2.destroyAllWindows()
print('Colleting Samples Complete!!!')