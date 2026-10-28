"""Reconstruct one reported configuration; default action only prints its config."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REVISION='a8539bee778d99e2902a1dd36aa157c6ebb3a42c'
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',type=Path,required=True)
parser.add_argument('--model',choices=['linear','mlp','cnn','rnn','transformer'],required=True)
parser.add_argument('--seed',type=int,choices=[69420,67,69],required=True)
parser.add_argument('--action',choices=['config','train','validation','test'],default='config')
parser.add_argument('--data-root',type=Path,default=Path('data'))
parser.add_argument('--output-root',type=Path,default=Path('runs/a1_reproduction'))
parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
parser.add_argument('--checkpoint',type=Path)
args=parser.parse_args()
repo=args.repo_root.resolve()
revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
if revision!=REVISION:
    raise SystemExit(f'Expected pinned source {REVISION}, got {revision}')
cfg=json.loads((ROOT/'evidence/configs'/f'{args.model}_seed{args.seed}.json').read_text(encoding='utf-8'))
cfg['data']['root']=str(args.data_root.resolve())
cfg['data']['split_file']=str(repo/'configs/a1/splits/fashion_mnist_split.json')
cfg['run']['output_root']=str(args.output_root.resolve())
cfg['run']['device']=args.device
cfg['_sources']={'model_config':str(repo/'configs/a1/models'/f'{args.model}.py'),
                 'protocol_config':str(repo/'configs/a1/protocol.py')}
if args.action=='config':
    print(json.dumps(cfg,indent=2)); raise SystemExit(0)
if args.action in ('validation','test') and args.checkpoint is None:
    parser.error('--checkpoint is required for evaluation; use trusted checkpoints only')
checkpoint=args.checkpoint.resolve() if args.checkpoint else None
os.chdir(repo)
sys.path.insert(0,str(repo/'src'))
from dlbench.common.config import validate_config
validate_config(cfg,strict=True)
if args.action=='train':
    from dlbench.a1.trainer import fit
    result=fit(cfg)
    print(result.run_dir)
else:
    from dataclasses import asdict
    from dlbench.a1.trainer import evaluate_checkpoint
    result=evaluate_checkpoint(cfg,checkpoint,split=args.action)
    print(json.dumps(asdict(result.metrics),indent=2))
