"""Extract only the selected RealSign classes using the supplied dataset splits."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import cv2
from features import create_detector,extract,ROOT
p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('--train-limit',type=int,default=300);p.add_argument('--test-limit',type=int,default=200);args=p.parse_args()
xs=[];ys=[];names=[];splits=[];summary={};rng=np.random.default_rng(42);seen=set();records=[]
with create_detector() as detector:
 for folder,split,limit in [('Training','train',args.train_limit),('Validation','validation',100),('Testing','test',args.test_limit)]:
  for label in ['A','B','C','D','L','V','Y']:
   files=sorted(Path(args.dataset,folder,label).glob('*.jpg'));rng.shuffle(files)
   ok=0;attempted=0;duplicates=0
   for f in files[:limit]:
    digest=hashlib.sha256(f.read_bytes()).hexdigest()
    if digest in seen:duplicates+=1;continue
    seen.add(digest);attempted+=1
    image=cv2.imread(str(f))
    if image is None:continue
    x,hands=extract(detector,cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    if x is not None:
     xs.append(x);ys.append(label);names.append(str(f));splits.append(split);ok+=1
     records.append({'path':str(f.relative_to(Path(args.dataset))),'sha256':digest,'label':label,'split':split,'hands':len(hands)})
   summary[split+'/'+label]={'attempted':attempted,'detected':ok,'exact_duplicates_skipped':duplicates}
   print(split,label,summary[split+'/'+label],flush=True)
np.savez_compressed(ROOT/'models/features.npz',x=xs,y=ys,names=names,split=splits)
(ROOT/'models/extraction_report.json').write_text(json.dumps(summary,indent=2))
(ROOT/'models/data_manifest.json').write_text(json.dumps(records,indent=2))
