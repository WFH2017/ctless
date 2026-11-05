"""Fixed-case 20-update diagnostic; metrics use the main trainer's body mask and ranges."""
import json,time,torch
from pathlib import Path
from train_true_unroll_siemens_nii import *
from nmrecon_training_projector import NMReconDeviceOperator

torch.manual_seed(42); device=torch.device('cuda:6');torch.cuda.set_device(device)
ds=SiemensNiiFullUnrollDataset(str(Path(SPLITS_DIR)/'siemens_nii_train.json'),str(Path(CTLESS_DIR)/'data/cache_siemens_nii'))
batch=collate_fn([ds[0]]);batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
model=EndToEndPhysicsUnrollSPECTNet(num_stages=4,base_ch=16,share_weights=True,net_depth=4).to(device).train()
criterion=TrueEndToEndSPECTLoss(poisson_weight=1e-3).to(device)
optim=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
root=Path(CTLESS_DIR)/'results/nmrecon_overfit20';root.mkdir(parents=True,exist_ok=True)
counts={}; times={}
def instrument(name,fn):
 def call(*args,**kw):
  torch.cuda.synchronize(device);t=time.perf_counter();r=fn(*args,**kw);torch.cuda.synchronize(device)
  counts[name]=counts.get(name,0)+1;times[name]=times.get(name,0.)+time.perf_counter()-t
  return r
 return call
for name in ['__init__','forward','backproject']:
 setattr(NMReconDeviceOperator,name,instrument(name,getattr(NMReconDeviceOperator,name)))
records=[]
for step in range(21):
 counts.clear();times.clear();optim.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats(device)
 torch.cuda.synchronize(device);start=time.perf_counter()
 projector=projector_from_batch(batch,str(device))
 out=model(lam_init_p=batch['lam_init_p'],lam_init_s=batch['lam_init_s'],proj_primary=batch['proj_primary'],scatter_est=batch['scatter_est'],projector_factory=projector,cond=batch['cond'],x_init_nac=batch['x_init_nac'],use_checkpoint=True,compute_projection_loss=True)
 losses=criterion(out,batch['x_gt'],batch['mu_gt'],batch['support_mask'])
 assert torch.isfinite(losses['loss']), 'nonfinite loss'
 with torch.no_grad():
  mask=batch['support_mask']>0.5
  error=(out['activity_final']-batch['x_gt']).square()
  muerr=(out['mu_final']-batch['mu_gt']).square()
  metrics=dict(masked_act_psnr=float(10*torch.log10(batch['x_gt'][mask].max().square()/error[mask].mean().clamp_min(1e-10))),masked_mu_psnr=float(10*torch.log10(0.25**2/muerr[mask].mean().clamp_min(1e-10))),body_sse=float(error[mask].sum()),air_sse=float(error[~mask].sum()),body_mse=float(error[mask].mean()),air_mse=float(error[~mask].mean()))
 if step<20:
  losses['loss'].backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optim.step()
 else: grad=torch.tensor(0.)
 torch.cuda.synchronize(device)
 record=dict(updates_before_forward=step,optimizer_update=step<20,case=batch['patient_id'],loss=float(losses['loss']),loss_act=float(losses['activity_rel_l1']),loss_mu=float(losses['atten_loss']),loss_poisson=float(losses['poisson_loss']),grad_before_clip=float(grad),seconds=time.perf_counter()-start,calls=dict(counts),operator_seconds=dict(times),peak_mb=torch.cuda.max_memory_allocated(device)/2**20,**metrics)
 records.append(record);print(json.dumps(record),flush=True)
 (root/'records.json').write_text(json.dumps(records,indent=2))
 del out,losses,projector
 record['allocated_after_release_mb']=torch.cuda.memory_allocated(device)/2**20
 (root/'records.json').write_text(json.dumps(records,indent=2))
torch.save({'model':model.state_dict(),'case':batch['patient_id'],'updates':20},root/'checkpoint_20steps.pt')
print('COMPLETED 20 updates plus final evaluation',flush=True)
