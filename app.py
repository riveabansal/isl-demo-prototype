from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import argparse,json,io,time,base64
import numpy as np
from PIL import Image,UnidentifiedImageError
from inference import Recognizer
ROOT=Path(__file__).resolve().parent
RECOGNIZER=None
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT/'static'),**kw)
    def reply(self,data,code=200):
        payload=json.dumps(data).encode();self.send_response(code)
        self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)))
        self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
    def do_GET(self):
        if self.path=='/api/info':
            report=json.loads((ROOT/'models/isl_mlp.json').read_text())
            return self.reply({'labels':RECOGNIZER.labels,'model':report['model'],
                'test_accuracy':report['test_accuracy'],'splits':report['splits'],
                'test_feature_coverage':report.get('test_feature_coverage'),'pipeline_correct_prediction_rate':report.get('pipeline_correct_prediction_rate'),'threshold':RECOGNIZER.bundle['threshold'],'source':'RealSign ISL alphabet dataset',
                'limitations':report['limitations']})
        return super().do_GET()
    def do_POST(self):
        if self.path!='/api/predict':self.send_error(404);return
        # Browser webcam frames are accepted from this local page only.
        origin=self.headers.get('Origin')
        if origin and origin not in [f'http://localhost:{self.server.server_port}',f'http://127.0.0.1:{self.server.server_port}']:
            return self.reply({'error':'Origin not permitted'},403)
        try:
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<=1500000:raise ValueError('Invalid image size')
            payload=json.loads(self.rfile.read(size));encoded=payload['image']
            if not isinstance(encoded,str):raise ValueError('Expected base64 image')
            raw=base64.b64decode(encoded,validate=True)
            with Image.open(io.BytesIO(raw)) as im:
                if im.width*im.height>1920*1080:raise ValueError('Frame is too large')
                rgb=np.asarray(im.convert('RGB'))
            start=time.perf_counter();result=RECOGNIZER.predict(rgb)
            result['latency_ms']=round((time.perf_counter()-start)*1000)
            self.reply(result)
        except (ValueError,KeyError,TypeError,UnidentifiedImageError):self.reply({'error':'Invalid image request'},400)
        except Exception as e:
            print('Inference error:',type(e).__name__,str(e),flush=True)
            self.reply({'error':'Inference failed; inspect server output'},500)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8001);a=p.parse_args()
    RECOGNIZER=Recognizer()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    print(f'ISL live recognizer: http://localhost:{a.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();RECOGNIZER.detector.close()
