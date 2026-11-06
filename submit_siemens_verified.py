from pathlib import Path
import datetime,shutil,subprocess,os,json,hashlib,socket
root=Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0]=='gn1'
out=root/'runs'/('siemens_nmrecon_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
code=out/'code';code.mkdir(parents=True)
for p in root.glob('*.py'):shutil.copy2(p,code/p.name)
shutil.copytree(root/'models',code/'models',ignore=shutil.ignore_patterns('__pycache__'))
for name in ['splits_siemens_nii','splits_ge_nii']:shutil.copytree(root/name,code/name)
(code/'data').symlink_to(root/'data',target_is_directory=True)
nm=code/'nmrecon_autograd';nm.mkdir();(nm/'build').mkdir()
for p in (root/'nmrecon_autograd').glob('*.py'):shutil.copy2(p,nm/p.name)
for p in ['library_manifest.json','build/libnmrecon.so']:shutil.copy2(root/'nmrecon_autograd'/p,nm/p)
python='/home/wengfenghua/anaconda3/envs/hqmira/bin/python'
cmd=[python,'-u',str(code/'train_true_unroll_siemens_nii.py'),'--epochs','50','--num-stages','4','--gpu-id','0','--lr','2e-4','--weight-decay','1e-4','--grad-accum-steps','4','--num-workers','4','--projector-type','nmrecon','--poisson-weight','0.001','--pretrained-mu-ckpt','','--output-dir',str(out)]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
with (out/'console.log').open('w') as log:
 p=subprocess.Popen(cmd,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
manifest=dict(host=socket.gethostname(),pid=p.pid,physical_gpu=1,command=cmd,initialization='from scratch',library_sha256=hashlib.sha256((nm/'build/libnmrecon.so').read_bytes()).hexdigest(),code_sha256={str(f.relative_to(code)):hashlib.sha256(f.read_bytes()).hexdigest() for f in code.rglob('*.py')})
(out/'submission.json').write_text(json.dumps(manifest,indent=2));(out/'train.pid').write_text(str(p.pid));(root/'latest_siemens_submission.txt').write_text(str(out));print(json.dumps({'pid':p.pid,'output':str(out)}),flush=True)
