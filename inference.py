import threading
import numpy as np
import torch
from features import create_detector,extract,ROOT
from model import LandmarkMLP

class Recognizer:
    def __init__(self):
        torch.set_num_threads(2)
        self.bundle=torch.load(ROOT/'models/isl_mlp.pt',map_location='cpu',weights_only=True)
        self.labels=self.bundle['labels'];self.net=LandmarkMLP(len(self.labels))
        self.net.load_state_dict(self.bundle['state_dict']);self.net.eval()
        self.detector=create_detector();self.lock=threading.Lock()
    def predict(self,rgb):
        with self.lock:
            features,hands=extract(self.detector,rgb)
        if features is None:return {'label':None,'status':'No hands detected','score':None,'hands':[],'top':[]}
        x=(torch.tensor(features)-self.bundle['mean'])/self.bundle['std']
        with torch.no_grad():probs=self.net(x[None]).softmax(1)[0].numpy()
        order=np.argsort(probs)[::-1];idx=int(order[0]);score=float(probs[idx])
        distance=float(torch.linalg.vector_norm(x-self.bundle['centers'][idx]))
        radius=float(self.bundle['radii'][idx])
        accepted=score>=self.bundle['threshold'] and distance<=max(radius*1.25,4.) and len(hands)==self.bundle['expected_hands'][self.labels[idx]]
        return {'label':self.labels[idx] if accepted else None,'candidate':self.labels[idx],
                'status':'Prediction available' if accepted else 'Uncertain / outside training range',
                'score':score,'hands':hands,'distance':distance,
                'top':[{'label':self.labels[int(i)],'score':float(probs[i])} for i in order[:3]]}
