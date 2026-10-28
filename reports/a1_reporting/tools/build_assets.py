"""Build report tables and scientific figures from saved A1 artifacts; no model execution."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics as st
import struct
from datetime import datetime, timezone, timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.datasets import FashionMNIST
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--train-root',type=Path,default=Path('C:/Users/tuank/Downloads/dlbench_a1_main/dlbench_a1_main'))
parser.add_argument('--evaluation-root',type=Path,default=Path('C:/Users/tuank/Downloads/cpu-0bd6ba8e08e9/cpu-0bd6ba8e08e9'))
parser.add_argument('--timing-root',type=Path,default=PROJECT/'analysis/a1-timing-only-2026-10-24/local_cpu_validation/cpu-f7208f3550fc')
parser.add_argument('--raw-root',type=Path,default=Path('F:/Github/deep-learning-benchmark-suite/data/FashionMNIST/raw'))
parser.add_argument('--code-root',type=Path,default=Path('F:/Github/deep-learning-benchmark-suite-colab'))
args=parser.parse_args()
FIG=ROOT/'figures'; TAB=ROOT/'tables'; EVID=ROOT/'evidence'
for p in [FIG,TAB,EVID,EVID/'configs']:
    p.mkdir(parents=True,exist_ok=True)
models=['linear','mlp','cnn','rnn','transformer']; names=['Linear','MLP','CNN','GRU','Transformer']
labels=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
seeds=[69420,67,69]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
rows=lambda p:list(csv.DictReader(p.open(encoding='utf-8',newline='')))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def writejson(name,data):
    (EVID/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def csvout(name,data):
    with (EVID/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
def ms(values,digits=3,scale=1):
    a=[float(x)*scale for x in values]
    return f'{st.mean(a):.{digits}f} $\\pm$ {st.stdev(a):.{digits}f}'
def table(name,alignment,header,data):
    body='\\begin{tabular}{'+alignment+'}\n\\toprule\n'+header+r' \\'+'\n\\midrule\n'
    body+='\n'.join(' & '.join(map(str,row))+r' \\' for row in data)
    body+='\n\\bottomrule\n\\end{tabular}\n'
    (TAB/name).write_text(body,encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'axes.labelsize':11,
                     'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':9,'savefig.dpi':200})
def savefig(fig,name):
    fig.savefig(FIG/name,bbox_inches='tight',facecolor='white'); plt.close(fig)

# Canonical cached IDX data, used for post-hoc report figures only.
raw_identity=[]
for filename,expected_md5 in FashionMNIST.resources:
    p=args.raw_root/filename
    assert hashlib.md5(p.read_bytes()).hexdigest()==expected_md5,filename
    unpacked=args.raw_root/filename.removesuffix('.gz')
    assert gzip.decompress(p.read_bytes())==unpacked.read_bytes()
    raw_identity.append({'file':filename,'gzip_bytes':p.stat().st_size,'idx_bytes':unpacked.stat().st_size,
                         'published_md5':expected_md5,'sha256':sha(p),'idx_sha256':sha(unpacked)})
def idx(name):
    data=(args.raw_root/name).read_bytes(); magic=struct.unpack('>I',data[:4])[0]
    if magic==2051:
        n,h,w=struct.unpack('>III',data[4:16]); return np.frombuffer(data,dtype=np.uint8,offset=16).reshape(n,h,w)
    assert magic==2049
    n=struct.unpack('>I',data[4:8])[0]; return np.frombuffer(data,dtype=np.uint8,offset=8,count=n)
train_images=idx('train-images-idx3-ubyte'); train_labels=idx('train-labels-idx1-ubyte')
test_images=idx('t10k-images-idx3-ubyte'); test_labels=idx('t10k-labels-idx1-ubyte')
split=read(args.code_root/'configs/a1/splits/fashion_mnist_split.json')
indices={k:np.asarray(split[k+'_indices']) for k in ['train','validation','test']}
assert not set(indices['train']) & set(indices['validation'])
assert set(indices['train'])|set(indices['validation'])==set(range(60000))
counts={s:np.bincount((test_labels if s=='test' else train_labels)[ids],minlength=10).tolist() for s,ids in indices.items()}
assert counts=={'train':[5000]*10,'validation':[1000]*10,'test':[1000]*10}
writejson('raw_data_identity.json',{'files':raw_identity,'partition_class_counts':counts,'split_seed':split['split_seed'],
                                   'images_shape':[list(train_images.shape),list(test_images.shape)],
                                   'use':'Post-hoc report figures; no training or model inference.'})
inventory=[[r['file'].replace('-idx3-ubyte.gz',' images').replace('-idx1-ubyte.gz',' labels'),
            f"{r['idx_bytes']:,}",f"{r['gzip_bytes']:,}"] for r in raw_identity]
inventory.append(['Total',f"{sum(r['idx_bytes'] for r in raw_identity):,}",f"{sum(r['gzip_bytes'] for r in raw_identity):,}"])
table('data_inventory.tex','lrr',r'Canonical resource & IDX bytes & Gzip bytes',inventory)
table('class_counts.tex','rlrrr',r'ID & Class & Train & Validation & Test',
      [[i,label,5000,1000,1000] for i,label in enumerate(labels)])
fig,axes=plt.subplots(2,5,figsize=(8,4.8),layout='constrained')
selected_training={}
for c,axis in enumerate(axes.flat):
    selected=[int(i) for i in indices['train'] if train_labels[i]==c][:4]; selected_training[labels[c]]=selected
    montage=np.full((58,58),255,dtype=np.uint8)
    for k,i in enumerate(selected):
        y,x=(k//2)*30,(k%2)*30; montage[y:y+28,x:x+28]=train_images[i]
    axis.imshow(montage,cmap='gray',vmin=0,vmax=255,interpolation='nearest'); axis.set_title(labels[c],fontsize=11); axis.axis('off')
savefig(fig,'training_examples.png'); writejson('eda_selection.json',{'selection':'First four saved training indices per class','sample_indices':selected_training})
fig,axis=plt.subplots(figsize=(7.4,3),layout='constrained'); x=np.arange(10)
for off,(s,color) in zip([-.24,0,.24],[('train','#306a9b'),('validation','#d08435'),('test','#4b8973')]):
    axis.bar(x+off,counts[s],width=.24,label=s.capitalize(),color=color)
axis.set_xticks(x,labels,rotation=35,ha='right'); axis.set_ylabel('Examples per class'); axis.legend(ncol=3)
savefig(fig,'class_distribution.png')

# All classification metrics and timing aggregates come from saved records.
records=[]; seed67={}; perclass=[]; source_hashes={}; timing_records=[]
for model,name in zip(models,names):
    for seed in seeds:
        run=next((args.train_root/'main').glob(f'a1_{model}_main_seed{seed}_*'))
        c=read(run/'config.json'); hist=rows(run/'history.csv'); best=read(run/'metrics.json')
        ev=args.evaluation_root/run.name; b=read(ev/'benchmark/benchmark.json')
        validation=read(ev/'validation/metrics.json'); test=read(ev/'test/metrics.json')
        cp_sha=sha(run/'best.pt'); assert validation['checkpoint_sha256']==test['checkpoint_sha256']==cp_sha
        timing=read(args.timing_root/'runs'/run.name/'benchmark.json')
        assert timing['checkpoint_sha256']==cp_sha and timing['status']=='completed'
        assert timing['measurement']['cpu_threads']==2 and timing['measurement']['batch_size']==64
        assert len(timing['measurement']['batch_ms_samples'])==100
        timing_records.append(timing)
        record={'model':model,'seed':seed,'run_id':run.name,'epochs':len(hist),'best_epoch':best['epoch']+1,
                'loop_minutes':sum(float(r['epoch_seconds']) for r in hist)/60,
                'total_parameters':b['parameter_count'],'trainable_parameters':b['trainable_parameters'],
                'checkpoint_sha256':cp_sha,**{'val_'+k:validation['metrics'][k] for k in ['loss','accuracy','macro_f1']},
                **{'test_'+k:test['metrics'][k] for k in ['loss','accuracy','macro_f1']}}
        records.append(record)
        shutil.copyfile(run/'config.json',EVID/'configs'/f'{model}_seed{seed}.json')
        for file in [run/'config.json',run/'history.csv',run/'metrics.json',run/'environment.json',
                     ev/'test/metrics.json',ev/'validation/metrics.json',ev/'test/per_class.csv',
                     args.timing_root/'runs'/run.name/'benchmark.json']:
            source_hashes[str(file)]=sha(file)
        for partition in ['validation','test']:
            for pc in rows(ev/partition/'per_class.csv'):
                perclass.append({'model':model,'seed':seed,'split':partition,'class_id':int(pc['class_id']),
                                  'class_name':pc['class_name'],'support':int(pc['support']),
                                  **{k:float(pc[k]) for k in ['precision','recall','f1']}})
        if seed==67:
            seed67[model]={'run':run,'eval':ev,'history':hist,'best':best,'record':record}
csvout('per_run.csv',records); csvout('per_class.csv',perclass)
quality=[]; main_rows=[]; parameter_rows=[]; training_rows=[]; timing_rows=[]; timing_summary=[]
original={(r['model'],r['eval_split']):r for r in read(args.evaluation_root/'summary/mean_std.json')}
for model,name in zip(models,names):
    group=[r for r in records if r['model']==model]
    for partition,key in [('validation','val'),('test','test')]:
        entry={'model':model,'split':partition,'n':3}
        for metric in ['accuracy','macro_f1','loss']:
            values=[r[key+'_'+metric] for r in group]
            entry[metric+'_mean']=st.mean(values); entry[metric+'_sample_sd']=st.stdev(values)
            assert math.isclose(entry[metric+'_mean'],original[model,partition][metric+'_mean'],abs_tol=1e-12)
            assert math.isclose(entry[metric+'_sample_sd'],original[model,partition][metric+'_sample_sd'],abs_tol=1e-12)
        quality.append(entry)
        main_rows.append([name,'Val.' if key=='val' else 'Test',ms([r[key+'_accuracy'] for r in group],3,100),
                          ms([r[key+'_macro_f1'] for r in group],3,100),ms([r[key+'_loss'] for r in group],5)])
    parameter_rows.append([name,f"{group[0]['total_parameters']:,}",f"{group[0]['trainable_parameters']:,}",0])
    training_rows.append([name,' / '.join(str(r['epochs']) for r in group),
                          ms([r['loop_minutes'] for r in group],2), ' / '.join(str(r['best_epoch']) for r in group)])
    ts=[r['measurement'] for r in timing_records if r['model']==model]
    summary={'model':model,'device':'Windows CPU','threads':2,'batch_size':64,'n':3}
    for key in ['median_batch_ms','amortized_ms_per_image','images_per_second']:
        summary[key+'_mean']=st.mean([r[key] for r in ts]); summary[key+'_sample_sd']=st.stdev([r[key] for r in ts])
    timing_summary.append(summary)
    timing_rows.append([name,ms([r['median_batch_ms'] for r in ts],3),ms([r['amortized_ms_per_image'] for r in ts],6),
                        ms([r['images_per_second'] for r in ts],0)])
table('performance.tex','llrrr',r'Model & Split & Accuracy (\%) & Macro-F1 (\%) & CE loss',main_rows)
table('parameters.tex','lrrr',r'Model & Total & Trainable & Frozen',parameter_rows)
table('training_time.tex','lcr c',r'Model & Epochs & Loop time (min) & Best epochs',training_rows)
table('inference.tex','lrrr',r'Model & Batch time (ms) & ms/image & Images/s',timing_rows)
csvout('quality_summary.csv',quality); csvout('timing_summary.csv',timing_summary)
writejson('timing_session.json',read(args.timing_root/'session.json'))
writejson('timing_per_run.json',timing_records)
shutil.copyfile(args.train_root/'run_ledger.json',EVID/'run_ledger.json')
shutil.copyfile(args.evaluation_root/'session.json',EVID/'evaluation_session.json')
shutil.copyfile(args.code_root/'AI_USAGE.md',EVID/'AI_USAGE_source.md')
writejson('report_facts.json',{'generated_at':datetime.now(timezone(timedelta(hours=7))).isoformat(),
    'quality':quality,'runs':records,'timing':timing_summary,'source_sha256':source_hashes,
    'no_training_or_checkpoint_execution':True,'raw_data_identity':'raw_data_identity.json'})

# Representative histories and seed-67 matrices/galleries from actual saved predictions.
for model,name in zip(models,names):
    item=seed67[model]; hist=item['history']; ev=item['eval']
    fig,axes=plt.subplots(1,2,figsize=(7.6,2.75),layout='constrained')
    for axis,metric in zip(axes,['loss','macro_f1']):
        for prefix,color,label in [('train','#306a9b','Train'),('val','#ce7634','Validation')]:
            axis.plot([int(r['epoch'])+1 for r in hist],[float(r[prefix+'_'+metric]) for r in hist],label=label,color=color,lw=1.25)
        axis.axvline(item['best']['epoch']+1,color='#777777',ls='--',lw=1,label='Selected epoch')
        axis.set(xlabel='Epoch (1-based)',ylabel='Cross-entropy' if metric=='loss' else 'Macro-F1'); axis.grid(alpha=.2); axis.legend(fontsize=8)
    savefig(fig,f'learning_{model}.png')
    payload=read(ev/'test/errors.json'); matrix=np.asarray(payload['confusion_counts'])
    fig,axis=plt.subplots(figsize=(6.8,5.15),layout='constrained')
    image=axis.imshow(matrix,cmap='Blues',vmin=0,vmax=1000)
    axis.set(xticks=range(10),yticks=range(10),xticklabels=labels,yticklabels=labels,
             xlabel='Predicted class',ylabel='True class',title=f'{name}: test, seed 67 (counts)')
    axis.tick_params(axis='x',rotation=45,labelsize=10)
    for i in range(10):
        for j in range(10):
            axis.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=10.5,color='white' if matrix[i,j]>500 else '#222222')
    fig.colorbar(image,ax=axis,fraction=.04,pad=.02)
    savefig(fig,f'confusion_{model}.png')
    data=[r for r in perclass if r['model']==model and r['seed']==67 and r['split']=='test']
    table(f'per_class_{model}.tex','lrrrr',r'Class & Precision & Recall & F1 & Support',
          [[r['class_name'],f"{r['precision']:.3f}",f"{r['recall']:.3f}",f"{r['f1']:.3f}",r['support']] for r in data])
    selected=read(ev/'test/gallery_selection.json')['samples']
    fig,axes=plt.subplots(3,3,figsize=(7.5,6.55),layout='constrained')
    for col,(kind,heading) in enumerate([('correct','Correct'),('difficult_correct','Difficult correct'),('errors','Confident error')]):
        for row,record in enumerate(selected[kind][:3]):
            sample_id,target,pred,confidence=record; source_id=int(sample_id.rsplit(':',1)[1])
            assert int(test_labels[source_id])==target
            axis=axes[row,col]; axis.imshow(test_images[source_id],cmap='gray',vmin=0,vmax=255,interpolation='nearest'); axis.axis('off')
            title=(heading+'\n' if row==0 else '')+f'ID {source_id}: {labels[target]}\nPred. {labels[pred]} ({confidence:.3f})'
            axis.set_title(title,fontsize=10.5,pad=5)
    savefig(fig,f'gallery_{model}.png')
    aggregate=[]
    for class_id,label in enumerate(labels):
        g=[r for r in perclass if r['model']==model and r['split']=='test' and r['class_id']==class_id]
        aggregate.append([label,ms([r['precision'] for r in g],2,100),ms([r['recall'] for r in g],2,100),ms([r['f1'] for r in g],2,100),1000])
    table(f'per_class_aggregate_{model}.tex','lrrrr',r'Class & Precision (\%) & Recall (\%) & F1 (\%) & Support',aggregate)

fig,axes=plt.subplots(1,2,figsize=(7.7,3.2),layout='constrained')
for axis,metric,title in zip(axes,['accuracy','macro_f1'],['Test accuracy','Test macro-F1']):
    ys=[next(r for r in quality if r['model']==m and r['split']=='test')[metric+'_mean']*100 for m in models]
    es=[next(r for r in quality if r['model']==m and r['split']=='test')[metric+'_sample_sd']*100 for m in models]
    axis.bar(names,ys,yerr=es,capsize=3,color='#306a9b'); axis.set_ylim(0,100); axis.set_ylabel('Percent'); axis.set_title(title)
    axis.tick_params(axis='x',rotation=30)
    for x,y in enumerate(ys): axis.text(x,y+2,f'{y:.2f}',ha='center',fontsize=9)
savefig(fig,'comparison.png')

# One regenerated, deterministic 64-example training batch for data illustration.
mu=.28585961086061534; sigma=.352790375749525
torch.manual_seed(69420)
pipeline=transforms.Compose([transforms.RandomCrop(28,padding=2,fill=0,padding_mode='constant'),transforms.ToTensor(),transforms.Normalize([mu],[sigma])])
ids=[int(i) for i in indices['train'][:64]]
batch=torch.stack([pipeline(Image.fromarray(train_images[i])) for i in ids])
fig,axes=plt.subplots(2,3,figsize=(7,4.9),layout='constrained')
for axis,i,tensor in zip(axes.flat,ids,batch):
    axis.imshow(tensor[0].numpy(),cmap='gray',vmin=-mu/sigma,vmax=(1-mu)/sigma,interpolation='nearest')
    axis.set_title(f'Train ID {i}: {labels[int(train_labels[i])]}',fontsize=11); axis.axis('off')
savefig(fig,'preprocessed_batch.png')
writejson('example_batch.json',{'scope':'Post-hoc regenerated data illustration; not a recorded training epoch batch',
    'seed':69420,'sample_ids':ids,'shape':list(batch.shape),'dtype':str(batch.dtype),'min':float(batch.min()),'max':float(batch.max()),
    'normalization':{'mean':mu,'std':sigma},'torch':torch.__version__})

seedrows=[]; idrows=[]
for model,name in zip(models,names):
    for r in [r for r in records if r['model']==model]:
        seedrows.append([name,r['seed'],r['epochs'],r['best_epoch'],f"{100*r['val_accuracy']:.2f}",f"{100*r['test_accuracy']:.2f}",f"{100*r['test_macro_f1']:.3f}",f"{r['test_loss']:.5f}"])
        idrows.append([name,r['seed'],r'\texttt{'+r['run_id'].replace('_',r'\_')+'}'])
table('seed_results.tex','lrrrrrrr',r'Model & Seed & Epochs & Best & Val. acc. & Test acc. & Test F1 & Test CE',seedrows)
table('run_ids.tex','lrl',r'Model & Seed & Run ID',idrows)
print(json.dumps({'runs':len(records),'quality_rows':len(quality),'per_class_rows':len(perclass),
                  'figures':len(list(FIG.glob('*.png'))),'training_performed':False,'checkpoint_execution':False,
                  'timing_scope':'Previously measured Windows CPU; no remeasurement'},ensure_ascii=False))
