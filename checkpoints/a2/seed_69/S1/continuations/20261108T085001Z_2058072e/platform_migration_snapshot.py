"""Explicit T4 cuda:0 continuation: preserve payload, adapt unused GPU RNG count.

The unchanged trainer still validates the complete new identity. Original parent
files are immutable; cross-platform bitwise reproducibility is not claimed.
"""
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import uuid

import a2_resume as recovery
import a2_support as support


def check_transition(previous,current):
    changes=recovery.differences(previous,current)
    allowed=set(recovery.HOST_ONLY_FIELDS)|{'environment.gpus','environment.python_version'}
    if any(x['field'] not in allowed for x in changes):
        raise ValueError('Numerical/config/source/data drift is blocked: '+', '.join(x['field'] for x in changes if x['field'] not in allowed))
    old,new=previous['environment'],current['environment']
    if old['python_version'].split('.')[:2]!=new['python_version'].split('.')[:2]:
        raise ValueError('Python major/minor must match the checkpoint; patch drift is recorded only.')
    a,b=old['gpus'],new['gpus']
    if not a or not b or len(b)>len(a) or len(a)>2:
        raise ValueError('Supported CUDA RNG mapping retains active GPU0 and can drop unused GPUs only.')
    for gpu in (a[0],b[0]):
        if 'T4' not in gpu['name'] or gpu['compute_capability']!='7.5':
            raise ValueError('Select a Colab T4 GPU to continue this T4 checkpoint.')
    for key in ('compute_capability','multi_processor_count','index'):
        if a[0][key]!=b[0][key]:raise ValueError('Active GPU0 architecture/index differs: '+key)
    if previous['device'] not in ('cuda','cuda:0') or current['device']!=previous['device']:
        raise ValueError('This continuation requires the original active cuda:0 device.')
    return changes


def map_cuda_rng(rng,old_count,new_count):
    import torch
    if not 1<=new_count<=old_count<=2 or len(rng['torch_cuda'])!=old_count:
        raise ValueError('Saved CUDA RNG topology does not match source/target GPUs')
    if any(not isinstance(x,torch.Tensor) or x.dtype!=torch.uint8 or x.ndim!=1 for x in rng['torch_cuda']):
        raise ValueError('Invalid saved CUDA RNG byte tensors')
    copied=dict(rng);copied['torch_cuda']=list(rng['torch_cuda'][:new_count])
    return copied


def load_state(path):
    import torch
    import numpy as np
    with torch.serialization.safe_globals([np._core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype('uint32'))]):
        return torch.load(path,map_location='cpu',weights_only=True)


def fork_transition(context,origin):
    import torch
    origin=Path(origin).resolve();source=context['run'].resolve()
    if source.name!='run' or not source.is_relative_to(origin):raise ValueError('Parent run escapes its origin')
    old=context['metadata'];current=context['identity'];changes=check_transition(old['identity'],current)
    if not changes:raise ValueError('No identity transition to record')
    if support.digest(old['identity'])!=old['identity_sha256'] or support.digest(current)!=context['fingerprint']:
        raise ValueError('Invalid source/target identity hash')
    last=load_state(source/'last.pt')
    if last['identity_sha256']!=old['identity_sha256'] or len(last['history'])!=last['epoch']:
        raise ValueError('Parent checkpoint identity/history invalid')
    if last['best']['epoch']==last['epoch']:
        best=last
    else:
        best=load_state(source/'best.pt')
        if best['epoch']!=last['best']['epoch'] or best['identity_sha256']!=old['identity_sha256']:
            raise ValueError('Parent best checkpoint does not match validation selection')
    if json.loads((source/'effective_config.json').read_text())!=current['config']:
        raise ValueError('Config must remain identical when continuing')
    inputs={n:support.sha256_file(source/n) for n in ('metadata.json','last.pt','best.pt','effective_config.json','history.json') if (source/n).is_file()}
    old_count=len(old['identity']['environment']['gpus']);new_count=len(current['environment']['gpus'])
    mapped_last=map_cuda_rng(last['rng'],old_count,new_count);mapped_best=map_cuda_rng(best['rng'],old_count,new_count)
    suffix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    target=origin/'continuations'/suffix;run=target/'run';run.mkdir(parents=True,exist_ok=False)
    for name in ('setup.json','frozen_config.json','pilot.json','manifest.json','data_checks.json','data_overlay.png','helper_snapshot.py'):
        if (origin/name).is_file():shutil.copy2(origin/name,target/name)
    if (source/'source').is_dir():shutil.copytree(source/'source',run/'source')
    platform=old.get('platform_profile','Kaggle')
    profile='hybrid_Kaggle_to_Colab' if platform.startswith(('Kaggle','hybrid')) else 'Colab'
    record={'kind':'documented_platform_continuation','origin_root':str(origin),'parent_run':str(source),
        'parent_run_id':old['run_id'],'parent_identity_sha256':old['identity_sha256'],
        'current_identity_sha256':context['fingerprint'],'resumed_epoch':last['epoch'],
        'differences':changes,'source_platform':platform,'target_platform':'Colab',
        'cuda_rng_mapping':{'active_source_index':0,'active_target_index':0,'old_count':old_count,'new_count':new_count,
                            'dropped_unused_source_indices':list(range(new_count,old_count)),
                            'active_rng_sha256':hashlib.sha256(mapped_last['torch_cuda'][0].numpy().tobytes()).hexdigest()},
        'original_input_sha256':inputs,'created_utc':datetime.now(timezone.utc).isoformat(),
        'migration_helper_sha256':support.sha256_file(Path(__file__)),
        'note':'Same model/optimizer/history/best/patience/config/source/tensor libraries and active T4 GPU0. '
               'Unused GPU RNG states may be dropped; Python patch/host/memory/topology differences recorded. '
               'A continuation of the same seed, not an independent repeat; no cross-platform bitwise guarantee.'}
    meta=deepcopy(old);meta.update(run_id=str(uuid.uuid4()),identity=deepcopy(current),identity_sha256=context['fingerprint'],
        environment=deepcopy(current['environment']),dependencies=deepcopy(current['dependencies']),
        source=deepcopy(context['source']),continuation=record,platform_profile=profile)
    meta['experiment_id']=old.get('experiment_id',old['run_id'])
    support.write_json(run/'metadata.json',meta);support.write_json(run/'effective_config.json',current['config'])
    for name,state,rng in (('last.pt',last,mapped_last),('best.pt',best,mapped_best)):
        child=dict(state);child['identity_sha256']=context['fingerprint'];child['rng']=rng
        torch.save(child,run/name)
    support.write_json(run/'history.json',last['history'])
    support.write_json(run/'metrics.json',{'best':last['best'],'epochs_completed':last['epoch'],'kind':meta['kind']})
    support.write_json(target/'continuation.json',record);shutil.copy2(Path(__file__),target/'platform_migration_snapshot.py')
    for name,digest in inputs.items():
        if support.sha256_file(source/name)!=digest:raise ValueError('Parent changed while copying; pointer not published')
    support.write_json(origin/'resume_pointer.json',{'target_relative':target.relative_to(origin).as_posix(),
                                                  'identity_sha256':context['fingerprint']})
    return target


def train_with_migration(config,train,validation,origin):
    origin=Path(origin);target=recovery.resolve_target(origin);support.install_progress(target)
    try:return target,support.train_or_resume(config,train,validation,target/'run')
    except ValueError as error:
        if str(error)!='Resume config/source/device identity mismatch':raise
        context=recovery.mismatch_context(error)
        try:check_transition(context['metadata']['identity'],context['identity'])
        except ValueError as blocked:
            support.write_json(origin/'colab_resume_diagnostic.json',
                {'differences':recovery.differences(context['metadata']['identity'],context['identity']),
                 'blocked_reason':str(blocked)})
            raise
        target=fork_transition(context,origin);support.install_progress(target)
        print('Documented Colab continuation:',target,flush=True)
        return target,support.train_or_resume(config,train,validation,target/'run')
