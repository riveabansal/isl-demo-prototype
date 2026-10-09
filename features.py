from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'runtime'/'mpl'))
import cv2
import mediapipe as mp
import numpy as np
ROOT=Path(__file__).resolve().parent

def create_detector():
    return mp.tasks.vision.HandLandmarker.create_from_options(mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'models/hand_landmarker.task')),
        num_hands=2,min_hand_detection_confidence=.35,min_hand_presence_confidence=.35))

def encode(hands):
    # Sort by screen x; use one joint coordinate frame for relative hand geometry.
    hands=sorted(hands,key=lambda h:h[0][0])
    p=np.asarray(hands,dtype=np.float32)
    center=p[:,0,:2].mean(axis=0)
    scales=np.linalg.norm(p[:,9,:2]-p[:,0,:2],axis=1)
    scale=max(float(scales.mean()),.025)
    values=np.zeros((2,21,3),np.float32)
    values[:len(hands),:,:2]=(p[:,:,:2]-center)/scale
    values[:len(hands),:,2]=p[:,:,2]/scale
    return np.concatenate([values.ravel(),[float(len(hands)==1),float(len(hands)==2)]]).astype(np.float32)

def extract(detector,rgb):
    h,w=rgb.shape[:2]
    if max(h,w)<480:rgb=cv2.resize(rgb,(int(w*480/max(h,w)),int(h*480/max(h,w))))
    result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)))
    hands=[[[p.x,p.y,p.z] for p in hand] for hand in result.hand_landmarks]
    # Landmark x/z are width-normalized; put y in the same units.
    metric_hands=[[[x,y*rgb.shape[0]/rgb.shape[1],z] for x,y,z in hand] for hand in hands]
    return (encode(metric_hands) if hands else None),hands
