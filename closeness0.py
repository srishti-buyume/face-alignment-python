import mediapipe as mp
import numpy as np
import cv2
import os,glob,shutil
import yaml


def face_extractor(img):
    try:
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        h,w,_ = img.shape 
        face_detection_results = face_detection.process(img[:,:,::-1])
        if face_detection_results.detections:
            for face_no, face in enumerate(face_detection_results.detections):
                face_data = face.location_data
                faces =[[int(face_data.relative_bounding_box.xmin*w),int(face_data.relative_bounding_box.ymin*h),
                         int(face_data.relative_bounding_box.width*w),int(face_data.relative_bounding_box.height*h)]]
                for x, y, w, h in faces:
                    cropped_img = img[y-int(h/3):y + 13*int(h/12), x:x + w]
    except Exception as e:
        cropped_img = False
    return cropped_img
def face_existence(face):    
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)
    mp_drawing = mp.solutions.drawing_utils
    face_detection_results = face_detection.process(face[:,:,::-1])
    if face_detection_results.detections:
        for face_no, face in enumerate(face_detection_results.detections):
            if mp_face_detection.FaceKeyPoint(1).name == 'LEFT_EYE' or mp_face_detection.FaceKeyPoint(0).name == 'RIGHT_EYE':
                return True
    else:
        return False
    
def driver(img):
    ho,wo = img.shape[:2]
    try:
        face = face_extractor(img)
        hf,wf = face.shape[:2]
        face_check = face_existence(face)
        cond1 = int(((hf*wf)/(ho*wo))*100)
        if face_check:
            if cond1 > 50:
                return "Door hattja Bhai"
            elif cond1 < 10:
                return "Pass aaja Bhai"
            else:
                return "Ab Sahi hai" 
    except Exception as e:
        return "OOPS.... Face not found !"