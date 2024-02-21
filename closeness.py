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
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    face_detection_results = face_detection.process(face[:,:,::-1])
    if face_detection_results.detections:
        for face_no, face in enumerate(face_detection_results.detections):
            if mp_face_detection.FaceKeyPoint(1).name == 'LEFT_EYE' or mp_face_detection.FaceKeyPoint(0).name == 'RIGHT_EYE':
                return True
    else:
        return False

    
def landmark_face(face_cropped_img):
    list_landmarks_acne = []
    h,w,_ = face_cropped_img.shape
    with open('roi.yml') as f:
        config = list(yaml.safe_load_all(f))
    variable = config[0]
    acne = variable['face_points']
    points_acne = acne["landmarks"]
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.9)
        face_mesh_results = face_mesh_images.process(face_cropped_img[:,:,::-1])
        if face_mesh_results.multi_face_landmarks:
            for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
                for ind in points_acne:
                    list_landmark=[]
                    for pnt in ind:
                        list_landmark.append([int(face_landmarks.landmark[pnt].x*w),
                                              int(face_landmarks.landmark[pnt].y*h)])
                    list_landmarks_acne.append(list_landmark)
        print(list_landmarks_acne)
        if len(list_landmarks_acne)>0:
            flag = True
        else:
            flag=False
        return flag
    except Exception as e:
        print(e)
        return False

def driver(img):
    ho,wo = img.shape[:2]
    try:
        face = face_extractor(img)
        hf,wf = face.shape[:2]
        face_check = face_existence(face)
        landmark_check = landmark_face(face)
        
        cond1 = int(((hf*wf)/(ho*wo))*100)
        if face_check:
            if landmark_check==False:
                return "Not straight"
            elif cond1 > 50:
                return "Too close"
            elif cond1 < 10:
                return "Too far"
            else:
                return "Good" 
    except Exception as e:
        print(e)
        return "Face not found !"