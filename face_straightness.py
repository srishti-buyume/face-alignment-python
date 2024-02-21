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

def landmark_acne(face_cropped_img,points_acne):
    list_landmarks_acne = []
    h,w,_ = face_cropped_img.shape
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5)
        face_mesh_results = face_mesh_images.process(face_cropped_img[:,:,::-1])
        if face_mesh_results.multi_face_landmarks:
            for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
                for ind in points_acne:
                    list_landmark=[]
                    for pnt in ind:
                        list_landmark.append([int(face_landmarks.landmark[pnt].x*w),
                                              int(face_landmarks.landmark[pnt].y*h)])
                    list_landmarks_acne.append(list_landmark)
        return list_landmarks_acne
    except Exception as e:
        print("Error in landmark acne: ",e)
        return False

def driver(img):
    face = face_extractor(img)
    face_check = face_existence(face)
    d = {"right_face" : [[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,397, 365, 379, 378, 400, 377, 152]],
         "left_face" : [[148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,109,109]]}
    
    list_areas = []
    if face_check:
        for halve in list(d.values()):
            list_landmarks = landmark_acne(face,halve)
            try:
                area = cv2.contourArea(np.array(list_landmarks))
                list_areas.append(area)
            except Exception as e:
                pass
    if len(list_areas)>1:
        ratio = list_areas[0]/list_areas[1]
        if ratio<=1.5 and ratio>=0.5:
            return "Looking Straight"
    else:
        return "Not Looking Straight"
 
        