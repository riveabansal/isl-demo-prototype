"""Train and evaluate a reproducible landmark classifier. No pretrained sign weights."""
import argparse,json,copy
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from model import LandmarkMLP
ROOT=Path(__file__).resolve().parent

def train(features,checkpoint,epochs=300):
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4)
    loaded=np.load(features,allow_pickle=False)
    expected={'A':2,'B':2,'C':1,'D':2,'L':1,'V':1,'Y':2}
    mask=np.array([int(row[127]>.5)+1==expected[str(label)] for row,label in zip(loaded['x'],loaded['y'])])
    data={key:loaded[key][mask] for key in loaded.files}
    x=data['x'].astype(np.float32);labels=[str(v) for v in sorted(set(data['y']))]
    y=np.array([labels.index(v) for v in data['y']]);indices=np.arange(len(x))
    if 'split' in data:
        train_idx=indices[data['split']=='train'];val_idx=indices[data['split']=='validation'];test_idx=indices[data['split']=='test']
    else:
        train_idx,holdout=train_test_split(indices,test_size=.3,stratify=y,random_state=42)
        val_idx,test_idx=train_test_split(holdout,test_size=.5,stratify=y[holdout],random_state=42)
    if not all(len(v) for v in [train_idx,val_idx,test_idx]):raise ValueError('Training, validation, and test sets must be nonempty')
    raw_train_count=len(train_idx)
    mirrored=x[train_idx].copy()
    for row in mirrored:
        hands=row[:126].reshape(2,21,3);hands[:,:,0]*=-1
        if row[127]>.5:hands[:]=hands[::-1].copy()
    extra_ids=np.arange(len(x),len(x)+len(mirrored))
    x=np.concatenate([x,mirrored]);y=np.concatenate([y,y[train_idx]])
    train_idx=np.concatenate([train_idx,extra_ids])
    mean=x[train_idx].mean(0);std=x[train_idx].std(0);std=np.maximum(std,.1)
    x=(x-mean)/std
    tx=torch.from_numpy(x);ty=torch.from_numpy(y).long()
    net=LandmarkMLP(len(labels));opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    counts=np.bincount(y[train_idx],minlength=len(labels));weight=torch.tensor(len(train_idx)/(len(labels)*np.maximum(counts,1)),dtype=torch.float32)
    loss_fn=torch.nn.CrossEntropyLoss(weight=weight)
    best=float('inf');best_state=None;stale=0;curve=[]
    for epoch in range(epochs):
        net.train();order=np.random.permutation(train_idx)
        for i in range(0,len(order),64):
            ids=order[i:i+64];inputs=tx[ids]+torch.randn_like(tx[ids])*.025
            opt.zero_grad();loss=loss_fn(net(inputs),ty[ids]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():
            val_loss=loss_fn(net(tx[val_idx]),ty[val_idx]).item()
            acc=(net(tx[val_idx]).argmax(1)==ty[val_idx]).float().mean().item()
        curve.append({'epoch':epoch+1,'validation_loss':val_loss,'validation_accuracy':acc})
        if val_loss<best-.0001:best=val_loss;best_state=copy.deepcopy(net.state_dict());stale=0
        else:stale+=1
        if (epoch+1)%25==0:print(f'epoch {epoch+1}: val loss {val_loss:.3f}, val acc {acc:.3f}',flush=True)
        if stale>=45:break
    net.load_state_dict(best_state);net.eval()
    with torch.no_grad():
        logits=net(tx[test_idx]);prob=logits.softmax(1);pred=prob.argmax(1).numpy()
    # Reference feature distances support conservative rejection, not calibrated OOD detection.
    centers=np.stack([x[train_idx][y[train_idx]==i].mean(0) for i in range(len(labels))])
    radii=np.array([np.percentile(np.linalg.norm(x[train_idx][y[train_idx]==i]-centers[i],axis=1),98) for i in range(len(labels))])
    bundle={'state_dict':best_state,'labels':labels,'mean':torch.tensor(mean),'std':torch.tensor(std),
            'centers':torch.tensor(centers),'radii':torch.tensor(radii),'threshold':.75,'expected_hands':expected,'feature_version':'screen-order-128-v1'}
    torch.save(bundle,checkpoint)
    distances=np.linalg.norm(x[test_idx]-centers[pred],axis=1)
    accepted=(prob.max(1).values.numpy()>=.75)&(distances<=np.maximum(radii[pred]*1.25,4.))
    report={'model':'128 → 128 → 64 → N MLP','seed':42,'labels':labels,'counts':dict(zip(labels,[int(sum(data['y']==v)) for v in labels])),
            'splits':{'train':raw_train_count,'validation':len(val_idx),'test':len(test_idx)},'training_rows_with_mirror_augmentation':len(train_idx),
            'test_accuracy':float((pred==y[test_idx]).mean()),'accepted_test_coverage':float(accepted.mean()),'accepted_test_accuracy':float((pred[accepted]==y[test_idx][accepted]).mean()) if accepted.any() else None,'classification_report':classification_report(y[test_idx],pred,labels=list(range(len(labels))),target_names=labels,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(y[test_idx],pred,labels=list(range(len(labels)))).tolist(),'epochs_run':len(curve),'curve':curve,
            'limitations':['Image-level dataset splits are not verified signer-independent; adjacent or augmented images may inflate scores.','No live-camera or new-signer accuracy measured.','Static selected alphabet classes only; softmax score is not calibrated confidence.']}
    extraction_path=Path(features).with_name('extraction_report.json')
    if extraction_path.exists():
        extraction=json.loads(extraction_path.read_text())
        attempted=sum(v['attempted'] for k,v in extraction.items() if k.startswith('test/'))
        report['attempted_test_images']=attempted
        report['test_feature_coverage']=len(test_idx)/attempted
        report['pipeline_accepted_coverage']=float(accepted.sum()/attempted)
        report['pipeline_correct_prediction_rate']=float(((pred==y[test_idx])&accepted).sum()/attempted)
    report['filtered_missing_or_extra_hands']=int((~mask).sum())
    Path(checkpoint).with_suffix('.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(ROOT/'models/test_split.npz',indices=test_idx,names=data['names'][test_idx],labels=data['y'][test_idx])
    print(json.dumps({k:report[k] for k in ['labels','counts','splits','test_accuracy','epochs_run']},indent=2),flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',default=str(ROOT/'models/features.npz'));p.add_argument('--checkpoint',default=str(ROOT/'models/isl_mlp.pt'));p.add_argument('--epochs',type=int,default=300);a=p.parse_args();train(a.features,a.checkpoint,a.epochs)
