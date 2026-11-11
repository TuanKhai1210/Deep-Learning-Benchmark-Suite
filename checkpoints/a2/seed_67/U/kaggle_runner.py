"""Single-process Kaggle orchestration around the unchanged pinned A2 core."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import statistics
import time
import uuid
import zipfile

import a2_support as support
import a2_resume as recovery
import a2_test as testing
import a2_reference as reference

NAMES = ('U', 'S0', 'S1')
SEEDS = (67, 69)
EXPECTED = {'U': 1964369, 'S0': 3718513, 'S1': 3718513}
SOURCE_DIGEST = '13ea100cefed8e7bd13c38153ebd3321760804366e2b5a66b47d2076d7e0c503'
EXTRAS_PROTOCOL = {'batch_size': 4, 'warmup': 10, 'repeats': 50, 'gallery_count': 6,
                   'precision': 'float32', 'device': 'cuda:0'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


class SessionPause(Exception):
    pass


class Budget:
    def __init__(self, seconds, stop_file):
        self.deadline = time.monotonic() + seconds
        self.stop_file = Path(stop_file)

    def check(self, reserve=0):
        if self.stop_file.exists() or time.monotonic() + reserve >= self.deadline:
            raise SessionPause('Session budget/STOP file reached; saved epochs are retained.')


def runtime_profile():
    import torch
    from dlbench.common.reproducibility import collect_environment
    support.seed_for_unet(69420)
    env = deepcopy(collect_environment())
    for field in recovery.HOST_ONLY_FIELDS:
        parts = field.split('.')[1:]
        parent = env
        for key in parts[:-1]:
            parent = parent.get(key, {})
        parent.pop(parts[-1], None)
    return {'environment': env, 'dependencies': {p: importlib.metadata.version(p)
            for p in ('Pillow', 'transformers', 'huggingface-hub', 'tqdm', 'safetensors')},
            'cpu_threads': torch.get_num_threads(), 'default_dtype': str(torch.get_default_dtype())}


def install_verified_source_identity():
    """Revision is an upstream snapshot reference, explicitly not a live checkout."""
    import dlbench.a2.trainer as trainer
    if not hasattr(trainer, '_kaggle_original_source_identity'):
        trainer._kaggle_original_source_identity = trainer.source_identity
    original = trainer._kaggle_original_source_identity
    def verified():
        value = original()
        if value['source_digest'] != SOURCE_DIGEST:
            raise ValueError('Embedded core source has changed; preserve this suite.')
        value.update(revision=support.CODE_REVISION, status='verified_embedded_snapshot',
                     revision_kind='upstream_reference_for_embedded_files', live_git_checkout=False,
                     tracked_diff='')
        return value
    trainer.source_identity = verified


def prepare_bases(repo):
    from dlbench.a2.config import load_config, validate_config
    bases = {}
    for seed in SEEDS:
        for name, file in zip(NAMES, ('unet.py', 'segformer_off.py', 'segformer_on.py')):
            cfg = load_config(Path(repo) / 'configs/a2' / file)
            cfg['run'].update(name=name, seed=seed, color_jitter=name == 'S1')
            cfg['protocol']['cuda_determinism'] = deepcopy(support.CUDA_POLICY)
            cfg['budget'] = {'batch_size': 4, 'epochs': 40, 'patience': 8}
            validate_config(cfg)
            bases[f'{seed}/{name}'] = cfg
        assert_pair(bases[f'{seed}/S0'], bases[f'{seed}/S1'])
    return bases


def assert_pair(c0, c1):
    copies = [deepcopy(c0), deepcopy(c1)]
    for name, cfg in zip(('S0', 'S1'), copies):
        if cfg['run']['name'] != name or cfg['run']['color_jitter'] != (name == 'S1'):
            raise ValueError('Invalid paired treatment')
        cfg['run'].pop('name'); cfg['run'].pop('color_jitter')
        cfg['protocol'].pop('pilot_reference', None)
    if copies[0] != copies[1]:
        raise ValueError('S0/S1 differ beyond training color augmentation')


def restore_suite(destination, input_root, tag, explicit=''):
    """Copy one saved suite; never merge versions or modify read-only inputs."""
    destination, input_root = Path(destination), Path(input_root)
    if (destination / 'suite_setup.json').is_file():
        return None
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Nonempty output without suite_setup.json; preserve it and use another tag.')
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            if candidate.suffix.lower() != '.zip':
                raise ValueError('RESTORE_FROM must be a saved suite directory or ZIP')
            stage = destination.parent / ('.restore-' + uuid.uuid4().hex)
            stage.mkdir(parents=True)
            with zipfile.ZipFile(candidate) as z:
                for item in z.infolist():
                    p = Path(item.filename.replace('\\', '/'))
                    if p.is_absolute() or '..' in p.parts or any(':' in x for x in p.parts):
                        raise ValueError('Unsafe resume ZIP member')
                    if (item.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError('Resume ZIP symlink is unsupported')
                z.extractall(stage)
            found = list(stage.rglob('suite_setup.json'))
            if len(found) != 1:
                raise ValueError('Resume ZIP must contain exactly one suite ledger')
            candidate = found[0].parent
        if not (candidate / 'suite_setup.json').is_file():
            raise ValueError('RESTORE_FROM has no suite_setup.json')
    else:
        found = [p.parent for p in input_root.rglob('suite_setup.json')
                 if read(p).get('suite_tag') == tag]
        if len(found) > 1:
            raise ValueError('Multiple saved suites found. Attach only the latest output or set RESTORE_FROM.')
        if not found:
            return None
        candidate = found[0]
    if read(candidate / 'suite_setup.json').get('suite_tag') != tag:
        raise ValueError('Saved suite tag differs from SUITE_TAG')
    destination.mkdir(parents=True, exist_ok=True)
    receipt = {}
    for path in sorted(candidate.rglob('*')):
        if path.is_symlink():
            raise ValueError('Resume input symlinks are unsupported')
        if path.is_file():
            target = destination / path.relative_to(candidate)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            expected = support.sha256_file(path)
            if support.sha256_file(target) != expected:
                raise ValueError('Resume copy hash mismatch')
            receipt[path.relative_to(candidate).as_posix()] = expected
    support.write_json(destination / 'restore_receipt.json',
                       {'source': str(candidate), 'copied_at': now(), 'source_sha256': receipt})
    print('Restored saved output:', candidate, flush=True)
    return candidate


def pin_suite(root, bases, bundle_sha):
    expected = {'bases': bases, 'helper_bundle_sha256': bundle_sha,
                'source_digest': SOURCE_DIGEST, 'source_revision': support.CODE_REVISION,
                'numerical_runtime': runtime_profile(), 'extras_protocol': EXTRAS_PROTOCOL,
                'completed_seed69420_sha256': support.sha256_file(Path(root)/'completed_seed69420.json')}
    path = Path(root) / 'suite_setup.json'
    if path.exists():
        saved = read(path)
        if saved['settings'] != expected or saved['settings_sha256'] != support.digest(expected):
            support.write_json(Path(root) / 'suite_mismatch.json',
                               {'differences': recovery.differences(saved['settings'], expected)})
            raise ValueError('Suite settings/helper/numerical runtime changed. Restore matching GPU/image/dependencies, or use a NEW SUITE_TAG for a separate experiment profile. See suite_mismatch.json.')
        return saved
    saved = {'suite_tag': Path(root).name, 'suite_uuid': str(uuid.uuid4()),
             'created_utc': now(), 'settings': expected, 'settings_sha256': support.digest(expected)}
    support.write_json(path, saved)
    return saved


def acquire_data(cache, input_root, explicit=''):
    cache, input_root = Path(cache), Path(input_root)
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_file():
            raise ValueError('DATA_ARCHIVE does not exist')
        return support.acquire_archive(candidate, lambda _: None)
    for candidate in sorted(input_root.rglob('dataset.zip')):
        if candidate.stat().st_size == support.DATA_BYTES and support.sha256_file(candidate) == support.DATA_SHA256:
            print('Using verified Kaggle input archive:', candidate, flush=True)
            return candidate
    def download(destination):
        import gdown
        try:
            result = gdown.download(id=support.DATA_FILE_ID, output=str(destination), quiet=False)
            if not result:
                raise RuntimeError('Drive returned no file')
        except Exception as error:
            raise RuntimeError('Dataset Drive download unavailable on Kaggle. Attach the ORIGINAL dataset.zip as a private Kaggle input and Run all again; auto-detection verifies the exact EDA hash. No Colab/Drive mount is used.') from error
    return support.acquire_archive(cache / 'dataset.zip', download)


def datasets(data_root, manifest, cfg):
    from dlbench.a2.data import TrashCanDataset
    return tuple(TrashCanDataset(data_root, manifest, split, cfg['preprocessing'],
                 seed=cfg['run']['seed'], color_jitter=cfg['run']['color_jitter'])
                 for split in ('train', 'validation'))


def overlay(dataset, path):
    from PIL import Image, ImageDraw
    import numpy as np
    palette = np.array([[0,0,0],[230,25,75],[60,180,75],[255,225,25],[0,130,200],
            [245,130,48],[145,30,180],[70,240,240],[240,50,230],[210,245,60],
            [250,190,212],[0,128,128],[220,190,255],[170,110,40],[255,250,200],
            [128,0,0],[170,255,195]], dtype=np.uint8)
    canvas = Image.new('RGB', (1024, 544), 'white')
    item = dataset[0]; rgb = item['image'].numpy().transpose(1,2,0)
    cfg = dataset.preprocessing
    rgb = np.uint8(np.clip((rgb*np.array(cfg['std']) + cfg['mean'])*255,0,255))
    mask = item['mask'].numpy(); colors = palette[np.minimum(mask,16)]; colors[mask==255] = 128
    canvas.paste(Image.fromarray(rgb),(0,32)); canvas.paste(Image.blend(Image.fromarray(rgb),Image.fromarray(colors),.5),(512,32))
    ImageDraw.Draw(canvas).text((10,10),'Train RGB | semantic overlay; ignore pixels gray',fill='black')
    canvas.save(path)


def pin_runs(root, repo, manifest, data_root, bases, bundle_sha, helpers, budget):
    """Freeze all six remaining configurations before main/test."""
    configs = {}
    for key, base in bases.items():
        seed, name = key.split('/'); origin = Path(root) / ('seed_' + seed) / name
        origin.mkdir(parents=True, exist_ok=True)
        for helper in helpers:
            shutil.copy2(helper, origin / helper.name)
            if helper.stem == 'a2_support':
                shutil.copy2(helper, origin / 'helper_snapshot.py')
        shutil.copy2(manifest, origin / 'manifest.json')
        for filename in ('data_checks.json','data_overlay.png'):
            if (Path(root)/filename).is_file():
                shutil.copy2(Path(root)/filename, origin/filename)
        ledger = origin / 'setup.json'
        if ledger.is_file():
            frozen = support.pin_setup(ledger, base, None, bundle_sha)
        else:
            budget.check(reserve=300)
            train, _ = datasets(data_root, manifest, base)
            pilot = support.pilot(base, train, steps=20)
            support.write_json(origin / 'pilot.json', pilot)
            if pilot['parameter_counts'] != {'total_params': EXPECTED[name],
                    'trainable_params': EXPECTED[name], 'frozen_params': 0}:
                raise ValueError('Unexpected total/trainable/frozen parameter counts')
            frozen = support.pin_setup(ledger, base, pilot, bundle_sha)
        configs[key] = frozen['config']
        support.write_json(origin / 'frozen_config.json', frozen['config'])
    for seed in SEEDS:
        assert_pair(configs[f'{seed}/S0'], configs[f'{seed}/S1'])
    return configs


def completion(root):
    import torch
    root = recovery.resolve_target(root)
    if not (root / 'run/last.pt').is_file():
        return root, None
    meta, cfg = read(root / 'run/metadata.json'), read(root / 'run/effective_config.json')
    state = torch.load(root / 'run/last.pt', map_location='cpu', weights_only=False)
    try:
        reason = testing.assert_complete(state, cfg, meta)
    except ValueError as error:
        if not str(error).startswith('Main run is only at epoch '):
            raise
        reason = None
    history_path = root / 'run/history.json'
    if not history_path.is_file() or state['history'] != read(history_path):
        # last.pt is the core's authoritative epoch-boundary recovery state.
        support.write_json(history_path, state['history'])
    best_path = root / 'run/best.pt'
    if state['best']['epoch'] == state['epoch']:
        best = torch.load(best_path, map_location='cpu', weights_only=False) if best_path.is_file() else None
        if best is None or best['epoch'] < state['epoch']:
            # Same repair as core fit: last was saved immediately before best/history.
            from dlbench.a2.trainer import atomic_save
            atomic_save(best_path, state)
    elif not best_path.is_file():
        raise ValueError('Earlier selected best.pt is missing; preserve this run for diagnosis.')
    return root, reason


def check_initial_pair(origin, new_metadata=None):
    name = Path(origin).name
    if name not in ('S0','S1'):
        return
    other = Path(origin).parent / ('S0' if name == 'S1' else 'S1')
    other = recovery.resolve_target(other)
    path = other / 'run/metadata.json'
    current = Path(origin) / 'run/metadata.json'
    if path.is_file() and (new_metadata is not None or current.is_file()):
        left = new_metadata if new_metadata is not None else read(current)
        right = read(path)
        if left['initial_state_sha256'] != right['initial_state_sha256']:
            raise ValueError('Paired initial weights differ; stopped before further training.')


def train_one(cfg, train, validation, origin, suite_root, budget):
    import dlbench.a2.trainer as trainer
    origin = Path(origin)
    # An interrupted first epoch has no checkpoint; retain the attempt before restart.
    partial = origin / 'run'
    if partial.exists() and not (partial / 'last.pt').exists():
        target = origin / 'pre_epoch_attempts' / uuid.uuid4().hex / 'run'
        if not partial.resolve().is_relative_to(origin.resolve()) or not target.resolve().is_relative_to(origin.resolve()):
            raise ValueError('Invalid pre-epoch recovery path')
        target.parent.mkdir(parents=True); partial.rename(target)
        print('Preserved incomplete first-epoch attempt:', target, flush=True)
    original_save = trainer.save_json
    def save_and_pause(path, value):
        path = Path(path)
        if path.name == 'metadata.json' and path.parent.name == 'run':
            check_initial_pair(origin, value)
        original_save(path, value)
        if path.name == 'metrics.json' and path.parent.name == 'run':
            # Core has already atomically saved last.pt, best.pt and history.json.
            support.write_json(Path(suite_root) / 'latest_epoch.json',
                {'run': str(origin.relative_to(suite_root)), 'epoch': value['epochs_completed'], 'saved_at': now()})
            budget.check()
    trainer.save_json = save_and_pause
    try:
        budget.check(reserve=300)
        active, outcome = recovery.train_with_recovery(cfg, train, validation, origin, support)
        support.write_json(active / 'outcome.json', outcome)
        check_initial_pair(origin)
        return active
    finally:
        trainer.save_json = original_save


def evaluate_and_extras(active, data_root, manifest, cfg, budget, extras=True):
    import dlbench.a2.trainer as trainer
    from dlbench.a2.engine import run_epoch
    from dlbench.a2.trainer import evaluate_checkpoint
    from dlbench.a2.data import TrashCanDataset
    from dlbench.a2.metrics import SegmentationMetrics
    budget.check(reserve=1200)
    # Held-out replay/test is not another training-validation phase in epoch logs.
    trainer.run_epoch = run_epoch
    print('Evaluating/reusing the validation-selected best checkpoint:', active, flush=True)
    _, validation = datasets(data_root, manifest, cfg)
    vpath = active / 'validation_best.json'
    checkpoint = support.sha256_file(active / 'run/best.pt')
    vmarker = active / 'validation_complete.json'
    identity = {'checkpoint_sha256': checkpoint,
                'run_identity_sha256': read(active / 'run/metadata.json')['identity_sha256']}
    if vpath.exists() and vmarker.exists():
        v = read(vpath)
        record = read(vmarker)
        if (any(v[k] != x for k,x in identity.items()) or record['identity'] != identity
                or record['output_sha256'] != support.sha256_file(vpath)):
            raise ValueError('Existing validation cache differs from best checkpoint')
    else:
        v = evaluate_checkpoint(active / 'run', validation, device='cuda', workers=0)
        support.write_json(vpath, v)
        support.write_json(vmarker, {'identity': identity, 'output_sha256': support.sha256_file(vpath)})
    selected = read(active / 'run/history.json')[v['selected_epoch']-1]['validation']
    if v['metrics']['confusion_matrix'] != selected['confusion_matrix'] or not math.isclose(v['metrics']['loss'],selected['loss'],rel_tol=0,abs_tol=1e-7):
        raise ValueError('Best validation replay differs from saved training history')
    testing.run_test(active, data_root, manifest, device='cuda')
    if extras:
        budget.check(reserve=300)
        extras_for_run(active, data_root, manifest, cfg)
    curves(active)


def curves(active):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = read(active / 'run/history.json')
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for split in ('train','validation'):
        for ax, key in zip(axes,('loss','foreground_miou')):
            ax.plot([r['epoch'] for r in history],[r[split][key] for r in history],label=split)
    for ax, label in zip(axes,('CE + Dice loss','Foreground mIoU')):
        ax.set(xlabel='Epoch',ylabel=label); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(active / 'learning_curves.png',dpi=150); plt.close(fig)


def extras_for_run(active, data_root, manifest, cfg):
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    from dlbench.a2.models import build_model, count_parameters
    from dlbench.a2.data import TrashCanDataset
    from dlbench.common.reproducibility import collect_environment
    identity = {'checkpoint_sha256': support.sha256_file(active / 'run/best.pt'),
                'run_identity_sha256': read(active / 'run/metadata.json')['identity_sha256'],
                'protocol': EXTRAS_PROTOCOL}
    marker = active / 'extras_complete.json'
    if marker.is_file():
        saved = read(marker)
        if saved['identity'] != identity or any(support.sha256_file(active/p) != h for p,h in saved['output_sha256'].items()):
            raise ValueError('Benchmark/gallery cache changed; preserve for review')
        return
    ds = TrashCanDataset(data_root, manifest, 'test', cfg['preprocessing'],seed=cfg['run']['seed'],color_jitter=False)
    state = torch.load(active/'run/best.pt',map_location='cpu',weights_only=False)
    model = build_model(cfg['model'],pretrained=False).to('cuda').eval(); model.load_state_dict(state['model']); del state
    images = torch.stack([ds[i]['image'] for i in range(EXTRAS_PROTOCOL['batch_size'])]).to('cuda')
    timings = []
    with torch.inference_mode():
        for _ in range(EXTRAS_PROTOCOL['warmup']):
            model(images)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        for _ in range(EXTRAS_PROTOCOL['repeats']):
            torch.cuda.synchronize(); start=time.perf_counter(); model(images); torch.cuda.synchronize()
            timings.append((time.perf_counter()-start)*1000)
        bench={'identity':identity,'environment':collect_environment(),'parameter_counts':count_parameters(model),
               'batch_ms_samples':timings,'median_batch_ms':statistics.median(timings),
               'median_batch_ms_per_image':statistics.median(timings)/images.shape[0],
               'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
               'note':'FP32 cuda:0 forward-only, batch4, excludes data loading/transfers/postprocessing; amortized batch time per image, NOT batch-one latency.'}
        support.write_json(active/'inference_benchmark.json',bench)
        palette=np.array([[0,0,0],[230,25,75],[60,180,75],[255,225,25],[0,130,200],[245,130,48],
            [145,30,180],[70,240,240],[240,50,230],[210,245,60],[250,190,212],[0,128,128],
            [220,190,255],[170,110,40],[255,250,200],[128,0,0],[170,255,195]],dtype=np.uint8)
        indices=np.linspace(0,len(ds)-1,EXTRAS_PROTOCOL['gallery_count'],dtype=int).tolist()
        gallery=Image.new('RGB',(1536,544*len(indices)),'white'); ids=[]
        for row,index in enumerate(indices):
            item=ds[index]; image=item['image']; mask=item['mask'].numpy()
            pred=model(image.unsqueeze(0).to('cuda')).argmax(1)[0].cpu().numpy()
            rgb=np.uint8(np.clip((image.numpy().transpose(1,2,0)*cfg['preprocessing']['std']+cfg['preprocessing']['mean'])*255,0,255))
            gt=palette[np.minimum(mask,16)]; gt[mask==255]=128
            pc=palette[pred]; pc[mask==255]=128
            for column,panel in enumerate((rgb,gt,pc)):
                gallery.paste(Image.fromarray(panel),(column*512,row*544+32))
            identifier=int(item['image_id']); ids.append(identifier)
            ImageDraw.Draw(gallery).text((8,row*544+8),f'Test ID {identifier}: RGB | ground truth | prediction; gray=ignore',fill='black')
        gallery.save(active/'prediction_gallery.png')
        support.write_json(active/'gallery_selection.json',{'test_indices':indices,'image_ids':ids,
            'selection':'Six evenly spaced manifest-order test examples fixed for every model/seed; not a frequency estimate.'})
    outputs={p:support.sha256_file(active/p) for p in ('inference_benchmark.json','prediction_gallery.png','gallery_selection.json')}
    support.write_json(marker,{'identity':identity,'output_sha256':outputs})
    del model, images; gc.collect(); torch.cuda.empty_cache()


def summarize(root):
    root=Path(root); rows=[]; status=[]
    metrics=('foreground_miou','foreground_dice','trash_miou')
    for seed in SEEDS:
        for name in NAMES:
            origin=root/f'seed_{seed}'/name; active=recovery.resolve_target(origin)
            h=active/'run/history.json'; history=read(h) if h.is_file() else []
            item={'seed':seed,'run':name,'epochs_saved':len(history),'active_relative':active.relative_to(root).as_posix(), 'status':'pending'}
            if (active/'run/last.pt').is_file():
                _,reason=completion(origin); item['status']='trained' if reason else 'training_incomplete'
                history=read(h); item['epochs_saved']=len(history)
            p=active/'test_complete.json'
            if p.is_file():
                marker=read(p); result=read(active/'test_best.json')
                if item['status']!='trained':
                    raise ValueError('Summary refuses test artifacts for an incomplete training run')
                if marker['output_sha256']!=support.sha256_file(active/'test_best.json'):
                    raise ValueError('Summary refuses changed test output')
                if any(result[k]!=v for k,v in marker['identity'].items()) or result['split']!='test':
                    raise ValueError('Summary test identity changed')
                if marker['identity']['checkpoint_sha256']!=support.sha256_file(active/'run/best.pt'):
                    raise ValueError('Summary test checkpoint changed')
                item['status']='complete' if (active/'extras_complete.json').exists() else 'test_complete'
                row={'seed':seed,'run':name,'epochs_completed':len(history),'best_epoch':result['selected_epoch'],
                     **{k:result['metrics'][k] for k in metrics}}
                if (active/'inference_benchmark.json').is_file():
                    row['batch4_median_ms_per_image']=read(active/'inference_benchmark.json')['median_batch_ms_per_image']
                rows.append(row)
            status.append(item)
    aggregates=[]
    for name in NAMES:
        selected=[r for r in rows if r['run']==name]; n=len(selected)
        if n:
            agg={'run':name,'n_seeds':n,'expected_seeds':len(SEEDS),'complete_planned_seeds':n==len(SEEDS)}
            for k in metrics:
                values=[r[k] for r in selected]; agg[k+'_mean']=statistics.mean(values)
                agg[k+'_sample_sd']=statistics.stdev(values) if n>1 else None
            aggregates.append(agg)
    paired=[]
    for seed in SEEDS:
        chosen={r['run']:r for r in rows if r['seed']==seed}
        if {'S0','S1'}<=chosen.keys():
            paired.append({'seed':seed,**{k+'_S1_minus_S0':chosen['S1'][k]-chosen['S0'][k] for k in metrics}})
    paired_summary={'n_pairs':len(paired),'expected_pairs':len(SEEDS),'per_seed':paired}
    for k in metrics:
        values=[r[k+'_S1_minus_S0'] for r in paired]
        paired_summary[k+'_delta_mean']=statistics.mean(values) if values else None
        paired_summary[k+'_delta_sample_sd']=statistics.stdev(values) if len(values)>1 else None
    summary={'updated_utc':now(),'all_planned_test_complete':len(rows)==len(SEEDS)*len(NAMES),'expected_test_runs':len(SEEDS)*len(NAMES),'planned_seeds':list(SEEDS),'completed_test_runs':len(rows),
             'run_status':status,'per_run':rows,'aggregates':aggregates,'paired_augmentation':paired_summary,
             'score_units':'0-1; multiply by100 for percent and percentage-point deltas',
             'profile':'This Kaggle suite only. Do not silently pool previous Colab runs.'}
    summary['combined_with_completed_seed69420']=reference.write_combined(root,summary)
    support.write_json(root/'summary.json',summary)
    for file,data in [('per_run.csv',rows),('mean_std.csv',aggregates),('paired_deltas.csv',paired)]:
        if data:
            fields=list(dict.fromkeys(k for row in data for k in row))
            with (root/file).open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(data)
    return summary


def export_suite(root):
    root=Path(root)
    summary=summarize(root)
    files={p.relative_to(root).as_posix():support.sha256_file(p) for p in sorted(root.rglob('*'))
           if p.is_file() and p.name!='export_manifest.json'}
    support.write_json(root/'export_manifest.json',{'exported_utc':now(),'files_sha256':files})
    package=root.parent/(root.name+'_results.zip'); temporary=package.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(root.parent))
    os.replace(temporary,package)
    print('Test runs completed:',summary['completed_test_runs'],'/ 6 | Saved:',package,flush=True)
    return package


def main(settings_path):
    import torch
    from dlbench.a2.trainer import evaluate_checkpoint
    settings=read(settings_path)
    root=Path(settings['output_root']); root.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2); torch.set_default_dtype(torch.float32)
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a Kaggle GPU accelerator and Internet before Run all.')
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    install_verified_source_identity()
    restore_suite(root,settings['input_root'],root.name,settings['restore_from'])
    bases=prepare_bases(settings['repo'])
    root.mkdir(parents=True,exist_ok=True)
    reference.install_snapshot(root,settings['completed_reference'],bases)
    pin_suite(root,bases,settings['bundle_sha256'])
    elapsed=max(0,time.time()-settings['session_started_unix'])
    budget=Budget(settings['session_hours']*3600-elapsed,root/'STOP')
    try:
        budget.check(reserve=600)
        archive=acquire_data(settings['cache'],settings['input_root'],settings['data_archive'])
        data_root=support.extract_material(archive,Path(settings['cache'])/'extracted')
        manifest=Path(settings['manifest'])
        train,validation=datasets(data_root,manifest,bases[f'{SEEDS[0]}/U'])
        qa=support.verify_training_data(train,validation)
        support.write_json(root/'data_checks.json',qa); overlay(train,root/'data_overlay.png')
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=bases[f'{SEEDS[0]}/S0']['model']['checkpoint'],
            revision=bases[f'{SEEDS[0]}/S0']['model']['revision'],allow_patterns=['config.json','model.safetensors'])
        configs=pin_runs(root,settings['repo'],manifest,data_root,bases,settings['bundle_sha256'],
                         [Path(p) for p in settings['helpers']],budget)
        selected=list(configs) if settings.get('selected_runs') is None else settings['selected_runs']
        if not selected or len(set(selected))!=len(selected) or not set(selected)<=set(configs):
            raise ValueError('SELECT_RUNS must contain unique keys from the six-run plan')
        if not settings['run_main']:
            print('RUN_MAIN=False: data checks and pilots only; no main/test.',flush=True)
            return
        for key in selected:
            seed,name=key.split('/'); origin=root/('seed_'+seed)/name; cfg=configs[key]
            print('\n===== RUN',key,'=====',flush=True)
            active,reason=completion(origin)
            support.install_runtime_policy(cfg)
            if reason is None:
                train,validation=datasets(data_root,manifest,cfg)
                active=train_one(cfg,train,validation,origin,root,budget)
            else:
                print('Training already complete:',reason,'— no additional epochs.',flush=True)
            evaluate_and_extras(active,data_root,manifest,cfg,budget,extras=settings['run_extras'])
            export_suite(root)
            gc.collect(); torch.cuda.empty_cache()
        support.write_json(root/'session_status.json',{'state':'selected_runs_finished','updated_utc':now()})
    except SessionPause as error:
        print(str(error),flush=True)
        support.write_json(root/'session_status.json',{'state':'paused_at_saved_epoch','reason':str(error),'updated_utc':now()})
    except BaseException as error:
        support.write_json(root/'session_status.json',{'state':'error_or_interrupted','error_type':type(error).__name__,
                           'message':str(error),'updated_utc':now()})
        raise
    finally:
        export_suite(root)
        print('SAVE THE KAGGLE OUTPUT. For the next session attach this output via Add Input; keep SUITE_TAG, GPU type and environment image unchanged. Working files are not a guarantee of saved output.',flush=True)


if __name__=='__main__':
    import sys
    main(sys.argv[1])
