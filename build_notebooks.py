#!/usr/bin/env python3
"""Build traceable tables from complete per-case records; never synthesize results.
Input JSON: {"checkpoint_sha256": "...", "metric_definition": "...",
"records": [{"patient_id": "...", "method": "...", "metrics": {"act_psnr": 40.0}}]}
The split JSON must contain the exact held-out patient list under "cases".
"""
import argparse, csv, hashlib, json, math, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def summarize(records_path, split_path, output):
    data = json.loads(records_path.read_text())
    split = json.loads(split_path.read_text())
    ids = [str(c['patient_id']) for c in split['cases']]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Split is empty or has duplicate patients')
    checkpoint = data.get('checkpoint_sha256', '')
    if len(checkpoint) != 64 or any(c not in '0123456789abcdefABCDEF' for c in checkpoint):
        raise ValueError('A checkpoint SHA256 is required')
    if not data.get('metric_definition'):
        raise ValueError('Metric definition, including mask and normalization, is required')
    groups = {}
    for row in data['records']:
        method, pid = row['method'], str(row['patient_id'])
        patients = groups.setdefault(method, {})
        if pid in patients:
            raise ValueError(f'Duplicate patient {pid} for {method}')
        metrics = row['metrics']
        if not metrics or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in metrics.values()):
            raise ValueError(f'Invalid or missing metrics: {method}/{pid}')
        patients[pid] = metrics
    if not groups:
        raise ValueError('No measured records')
    rows = []
    for method, patients in groups.items():
        if set(patients) != set(ids):
            raise ValueError(f'{method}: patient list does not match the locked split')
        keys = set(next(iter(patients.values())))
        if any(set(m) != keys for m in patients.values()):
            raise ValueError(f'{method}: inconsistent metric coverage')
        for metric in sorted(keys):
            values = [patients[pid][metric] for pid in ids]
            rows.append(dict(method=method, metric=metric, n=len(values), mean=statistics.mean(values), sd=statistics.stdev(values) if len(values)>1 else '', median=statistics.median(values)))
    output.mkdir(parents=True, exist_ok=True)
    with (output/'measured_summary.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=['method','metric','n','mean','sd','median']); writer.writeheader();writer.writerows(rows)
    provenance = dict(records=str(records_path.resolve()), records_sha256=hashlib.sha256(records_path.read_bytes()).hexdigest(), split=str(split_path.resolve()), split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(), checkpoint_sha256=checkpoint, metric_definition=data['metric_definition'])
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False))
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'tables')
    args=parser.parse_args()
    rows=summarize(args.records,args.split,args.output)
    print(f'Generated {len(rows)} measured summary rows in {args.output}')

if __name__=='__main__': main()
