"""Reuse audited Colab seed69420 scores as data; never load/train its models."""
from pathlib import Path
from copy import deepcopy
import csv
import hashlib
import json
import math
import statistics

KEYS=('foreground_miou','foreground_dice','trash_miou')
SOURCE_DIGEST='13ea100cefed8e7bd13c38153ebd3321760804366e2b5a66b47d2076d7e0c503'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)


def comparable(config):
    cfg=deepcopy(config);cfg['run'].pop('seed')
    for key in ('status','approval_reference','pilot_reference'):cfg['protocol'].pop(key,None)
    return cfg


def install_snapshot(root,source,bases):
    root=Path(root);source=Path(source);value=read(source)
    if value['seed']!=69420 or set(value['runs'])!={'U','S0','S1'}:
        raise ValueError('Prior snapshot must contain the completed seed69420 U/S0/S1')
    for name,row in value['runs'].items():
        if row['source_digest']!=SOURCE_DIGEST:
            raise ValueError('Prior source digest differs from this experiment')
        expected=bases['67/'+name]
        if comparable(row['effective_config'])!=comparable(expected):
            raise ValueError('Prior seed69420 scientific config differs: '+name)
        if row['seed']!=69420 or row['runtime_profile']!='Colab_seed69420':
            raise ValueError('Unexpected completed-run seed/runtime label')
        if any(not math.isfinite(row[k]) or not 0<=row[k]<=1 for k in KEYS):
            raise ValueError('Invalid prior scores')
    target=root/'completed_seed69420.json'
    if target.is_file() and target.read_bytes()!=source.read_bytes():
        raise ValueError('Completed-seed snapshot changed; preserve the existing suite')
    if not target.is_file():target.write_bytes(source.read_bytes())
    return value


def rows_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def write_combined(root,kaggle_summary):
    root=Path(root);prior=read(root/'completed_seed69420.json')
    rows=[]
    for name in ('U','S0','S1'):
        p=prior['runs'][name]
        rows.append({k:p[k] for k in ('seed','run','runtime_profile','verified_date','epochs_completed','best_epoch')+KEYS})
    for row in kaggle_summary['per_run']:
        if row['seed'] not in (67,69):
            raise ValueError('Only seeds67/69 may come from new Kaggle training')
        rows.append({k:row[k] for k in ('seed','run','epochs_completed','best_epoch')+KEYS}|
                    {'runtime_profile':'Kaggle_remaining_67_69','verified_date':kaggle_summary['updated_utc']})
    if len({(r['seed'],r['run']) for r in rows})!=len(rows):
        raise ValueError('Duplicate model/seed would double-count a completed result')
    aggregates=[]
    for name in ('U','S0','S1'):
        selected=[r for r in rows if r['run']==name];n=len(selected)
        profiles=sorted({r['runtime_profile'] for r in selected})
        a={'run':name,'n_seeds':n,'expected_seeds':3,'complete_three_seeds':n==3,
           'runtime_profiles':' | '.join(profiles),'mixed_runtime_profiles':len(profiles)>1,
           'isolates_seed_variability':False}
        for key in KEYS:
            values=[r[key] for r in selected];a[key+'_mean']=statistics.mean(values)
            a[key+'_sample_sd']=statistics.stdev(values) if n>1 else None
        aggregates.append(a)
    paired=[]
    for seed in (69420,67,69):
        arms={r['run']:r for r in rows if r['seed']==seed}
        if {'S0','S1'}<=arms.keys():
            if arms['S0']['runtime_profile']!=arms['S1']['runtime_profile']:
                raise ValueError('A paired augmentation comparison requires one profile within seed')
            paired.append({'seed':seed,'runtime_profile':arms['S0']['runtime_profile'],
                           **{k+'_S1_minus_S0':arms['S1'][k]-arms['S0'][k] for k in KEYS}})
    result={'completed_total_test_runs':len(rows),'expected_total_test_runs':9,
            'all_three_seeds_test_complete':len(rows)==9,'completed_seed69420_reused':True,
            'seed69420_training_or_inference_repeated':False,'per_run':rows,'aggregates':aggregates,
            'paired_deltas':paired,'n_pairs':len(paired),
            'quality_scores_only':True,'runtime_caveat':
            'Seed69420 is retained from audited Colab results;67/69 are Kaggle runs. '
            'Combined mean/sample SD describes these recorded runs and does not isolate seed-only variability. '
            'Report the platform/library/GPU differences; do not pool inference timings across profiles.'}
    result['paired_augmentation_summary']={'n_pairs':len(paired),'expected_pairs':3}
    for key in KEYS:
        values=[p[key+'_S1_minus_S0'] for p in paired]
        result['paired_augmentation_summary'][key+'_delta_mean']=statistics.mean(values)
        result['paired_augmentation_summary'][key+'_delta_sample_sd']=statistics.stdev(values) if len(values)>1 else None
    write(root/'combined_three_seed_summary.json',result)
    rows_csv(root/'combined_three_seed_results.csv',rows)
    rows_csv(root/'combined_three_seed_mean_std.csv',aggregates)
    rows_csv(root/'combined_three_seed_paired_deltas.csv',paired)
    return result
