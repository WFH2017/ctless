import json,time,torch
from pathlib import Path
import train_true_unroll_siemens_nii as train
import models.true_unroll_spect as model_code
import nmrecon_training_projector as physics
from nmrecon_autograd import nmrecon_torch as native
root=Path(__file__).resolve().parent
for mod in [train,model_code,physics,native]:
 assert Path(mod.__file__).resolve().is_relative_to(root),mod.__file__
 print('MODULE',mod.__file__,flush=True)
print('LIBRARY',native.DEFAULT_LIB_PATH,flush=True)
torch.cuda.set_device(6);device='cuda:6';torch.manual_seed(42)
ds=train.SiemensNiiFullUnrollDataset(str(root/'splits_siemens_nii/siemens_nii_train.json'),str(root/'data/cache_siemens_nii'))
batch=train.collate_fn([ds[0]]);batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
model=model_code.EndToEndPhysicsUnrollSPECTNet(num_stages=4,base_ch=16,share_weights=True,net_depth=4).to(device).train()
projector=physics.projector_from_batch(batch,device)
counts={};op=projector.operator
for name in ['forward','backproject']:
 fn=getattr(op,name)
 def counted(*a,_fn=fn,_name=name,**kw):
  counts[_name]=counts.get(_name,0)+1
  return _fn(*a,**kw)
 setattr(op,name,counted)
results=[];previous=None
for w in [0.,.001]:
 model.zero_grad(set_to_none=True);counts.clear();torch.cuda.synchronize();start=time.perf_counter()
 out=model(lam_init_p=batch['lam_init_p'],lam_init_s=batch['lam_init_s'],proj_primary=batch['proj_primary'],scatter_est=batch['scatter_est'],projector_factory=projector,cond=batch['cond'],x_init_nac=batch['x_init_nac'],use_checkpoint=True,compute_projection_loss=True)
 before=dict(counts)
 loss=model_code.TrueEndToEndSPECTLoss(poisson_weight=w)(out,batch['x_gt'],batch['mu_gt'],batch['support_mask'],batch['proj_primary'],batch['scatter_est'],projector)
 assert counts==before,'Loss repeated forward projection'
 loss['loss'].backward();torch.cuda.synchronize();elapsed=time.perf_counter()-start
 act=torch.cat([p.grad.flatten() for n,p in model.named_parameters() if 'act_prox_net' in n and p.grad is not None])
 mu=torch.cat([p.grad.flatten() for n,p in model.named_parameters() if 'init_mu_net' in n and p.grad is not None])
 assert torch.isfinite(act).all() and torch.isfinite(mu).all() and act.norm()>0 and mu.norm()>0
 row=dict(weight=w,seconds=elapsed,calls=dict(counts),mu_grad_norm=float(mu.norm()),act_grad_norm=float(act.norm()),**{k:float(v) for k,v in loss.items()})
 if previous is not None:
  row['activity_gradient_delta_norm']=float((act-previous).norm());assert row['activity_gradient_delta_norm']>0
 previous=act.detach().clone();results.append(row);print(json.dumps(row),flush=True)
 del out,loss,act,mu
(root/'runtime_verification.json').write_text(json.dumps({'modules':[m.__file__ for m in [train,model_code,physics,native]],'library':json.loads((root/'nmrecon_autograd/library_manifest.json').read_text()),'case':batch['patient_id'],'results':results},indent=2))
print('PASS',flush=True)
