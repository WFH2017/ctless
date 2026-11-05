"""Stage 2 only: native raw projection equivalence, not autograd validation.

Writes only to the selected output directory. Validation patients only.
Acceptance thresholds fixed here before running clinical geometry checks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ORIGINAL = Path('/home/wengfenghua/nmrecon_raw/nmrecon0708/NMRecon/nmrecon/build/bin/nmrecon')
TOL = 1e-4  # relative L2; accommodates float32 CUDA accumulation and unit round trip


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(a, b):
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    err = float(np.linalg.norm(a.astype('float64')-b)/max(np.linalg.norm(a.astype('float64')),1e-12))
    return {'relative_l2':err,'passed':err <= TOL,'size':int(a.size)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--binary',type=Path,required=True)
    ap.add_argument('--gpu',default='5')
    ap.add_argument('--out',type=Path,default=ROOT/'validation_mu')
    args=ap.parse_args()
    args.binary=args.binary.resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    report={'scope':'float32 mu interface; no autograd certification',
            'relative_l2_tolerance':TOL,'original_sha256':fingerprint(ORIGINAL),
            'new_sha256':fingerprint(args.binary),'results':{}}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu)
    def run(binary, config, flags, work, name, fail=False):
        start=time.monotonic()
        p=subprocess.run([str(binary),str(config),*map(str,flags)],cwd=work,env=env,
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (work/f'{name}.log').write_text(p.stdout)
        if fail:
            assert p.returncode != 0, name+' accepted invalid input'
        elif p.returncode:
            raise RuntimeError(f'{name} failed ({p.returncode}); see {work/name}.log')
        return time.monotonic()-start
    for vendor,folder in [('ge','GE'),('siemens','Siemens')]:
        split=PROJECT/f'ctless/splits_{vendor}_nii/{vendor}_nii_val.json'
        cases=json.loads(split.read_text())['cases']
        config=next(PROJECT/f'data/{folder}/nii'/c['patient_id']/'recon.json' for c in cases
                    if (PROJECT/f'data/{folder}/nii'/c['patient_id']/'recon.json').exists())
        cfg=json.loads(config.read_text()); case=config.parent
        # Resolve existing source paths; all exports remain in isolated work directory.
        def paths(value):
            if isinstance(value,dict):return {k:paths(v) for k,v in value.items()}
            if isinstance(value,list):return [paths(v) for v in value]
            if isinstance(value,str) and value and (case/value).exists():return str((case/value).resolve())
            return value
        cfg=paths(cfg)
        work=args.out/vendor;work.mkdir(exist_ok=True)
        local=work/'config.json';local.write_text(json.dumps(cfg))
        size=cfg['ReconstructSetting']['output']['image_meta']['size']
        shape=tuple(reversed(size))  # raw x-fastest => NumPy [z,y,x]
        z,y,x=np.indices(shape,dtype=np.float32)
        image=(0.01+np.exp(-((x-size[0]*.32)**2+(y-size[1]*.57)**2+(z-size[2]*.43)**2)/130)).astype('float32')
        image.tofile(work/'x.bin')
        rec={'config_sha256':fingerprint(config),'validation_split_sha256':fingerprint(split),'shape_zyx':shape,'checks':{}}
        report['results'][vendor]=rec
        run(ORIGINAL,local,['--forward-project',work/'x.bin',work/'original.bin'],work,'original')
        run(args.binary,local,['--forward-project',work/'x.bin',work/'ct.bin','--export-mu-map-cm',work/'mu.bin'],work,'ct_export')
        read=lambda name:np.fromfile(work/name,dtype=np.float32)
        rec['checks']['original_vs_new_ct']=compare(read('original.bin'),read('ct.bin'))
        # Replace CT with a nonexistent file: neural mu input must completely bypass CT.
        cfg['ReconstructSetting']['corrections']['attenuation']['ct_image']=str(work/'MUST_NOT_READ_CT.nii.gz')
        predicted=work/'predicted_config.json';predicted.write_text(json.dumps(cfg))
        run(args.binary,predicted,['--forward-project',work/'x.bin',work/'pred.bin','--mu-map-cm',work/'mu.bin'],work,'predicted_mu')
        rec['checks']['ct_vs_float_mu']=compare(read('ct.bin'),read('pred.bin'))
        mu=read('mu.bin');assert mu.size==image.size and np.isfinite(mu).all() and mu.min()>=0
        rec['mu_cm_range']=[float(mu.min()),float(mu.max())]
        rec['projection_float_count']=int(read('ct.bin').size)
        # A deliberately different mu must affect output; zero matches native no-AC.
        np.zeros(image.size,dtype=np.float32).tofile(work/'zero_mu.bin')
        run(args.binary,predicted,['--forward-project',work/'x.bin',work/'zero.bin','--mu-map-cm',work/'zero_mu.bin'],work,'zero_mu')
        cfg['ReconstructSetting']['corrections']['attenuation']['ct_image']=None
        noac=work/'noac_config.json';noac.write_text(json.dumps(cfg))
        run(ORIGINAL,noac,['--forward-project',work/'x.bin',work/'original_noac.bin'],work,'original_noac')
        rec['checks']['zero_mu_vs_noac']=compare(read('original_noac.bin'),read('zero.bin'))
        rec['checks']['mu_changes_projection']={'passed':not np.allclose(read('pred.bin'),read('zero.bin'))}
        # Signed probe tests raw backprojection equivalence, not merely positive images.
        rng=np.random.default_rng(20261105)
        rng.standard_normal(read('ct.bin').size).astype('float32').tofile(work/'probe.bin')
        run(ORIGINAL,local,['--backproject',work/'probe.bin',work/'bp_original.bin'],work,'bp_original')
        run(args.binary,predicted,['--backproject',work/'probe.bin',work/'bp_mu.bin','--mu-map-cm',work/'mu.bin'],work,'bp_mu')
        rec['checks']['backprojection_equivalence']=compare(read('bp_original.bin'),read('bp_mu.bin'))
        for label,data in [('wrong_size',np.zeros(3,dtype='float32')),
                           ('negative',np.full(image.size,-.1,dtype='float32')),
                           ('nan',np.full(image.size,np.nan,dtype='float32'))]:
            data.tofile(work/'invalid.bin')
            run(args.binary,predicted,['--forward-project',work/'x.bin',work/'invalid_out.bin','--mu-map-cm',work/'invalid.bin'],work,label,fail=True)
            rec['checks']['reject_'+label]={'passed':True}
        report['passed']=all(c['passed'] for r in report['results'].values() for c in r['checks'].values())
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(vendor,json.dumps(rec['checks']),flush=True)
    assert report['passed'],'Numerical acceptance failed; do not proceed'


if __name__=='__main__':main()
