import sys,time,json,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'nmrecon_autograd'))
from nmrecon_torch import NMReconDeviceOperator
D=6;torch.cuda.set_device(D);torch.manual_seed(42)
root=Path(__file__).resolve().parents[1]
raw=torch.load(root/'ctless/data/cache_siemens_nii/601344948.pt',weights_only=False)
x=raw['lambda_gt'].cuda(D).float();mu=raw['mu_gt'].cuda(D).float();g=torch.randn((64,128,128),device=f'cuda:{D}')
results={};values={}
for name,lib in [('baseline','libnmrecon_baseline.so'),('fast','libnmrecon.so')]:
 torch.cuda.synchronize();t=time.perf_counter();op=NMReconDeviceOperator(str(root/'data/Siemens/nii/601344948/recon.json'),D,str(root/'nmrecon_autograd/build'/lib));torch.cuda.synchronize();create=time.perf_counter()-t
 torch.cuda.synchronize();t=time.perf_counter();f=op.forward(x,mu);torch.cuda.synchronize();ft=time.perf_counter()-t
 t=time.perf_counter();b=op.backproject(g,mu);torch.cuda.synchronize();bt=time.perf_counter()-t
 lhs=(f.double()*g.double()).sum();rhs=(x.double()*b.double()).sum()
 results[name]=dict(create_s=create,forward_s=ft,backward_s=bt,lhs=float(lhs),rhs=float(rhs),adjoint_relative=float(abs(lhs-rhs)/torch.maximum(abs(lhs),abs(rhs))))
 values[name]=(f,b);print(name,results[name],flush=True)
for i,name in enumerate(['forward','backward']):
 a,b=values['baseline'][i],values['fast'][i];results[name+'_relative_l2']=float((a-b).norm()/a.norm())
print(json.dumps(results,indent=2),flush=True)
(root/'ctless/results/nmrecon_fast_validation.json').write_text(json.dumps(results,indent=2))
assert results['forward_relative_l2']<1e-5
assert results['fast']['adjoint_relative']<1e-4
