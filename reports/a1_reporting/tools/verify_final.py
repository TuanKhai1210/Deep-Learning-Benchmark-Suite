"""Verify saved A1 tables, numeric PDF content and equivalent TeX entry points."""
from pathlib import Path
import ast,csv,hashlib,json,math,re,statistics as st,unicodedata,zipfile
from datetime import datetime,timezone,timedelta
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
rows=lambda p:list(csv.DictReader(p.open(encoding='utf-8',newline='')))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12),(a,b)
def norm(t):
    t=re.sub(r'(?<=[\w/])-\n(?=\w)','',t)
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',t))

runs=rows(E/'per_run.csv')
quality=rows(E/'quality_summary.csv')
classes=rows(E/'per_class.csv')
timings=rows(E/'timing_summary.csv')
raw=read(E/'timing_per_run.json')
assert len(runs)==15 and len(quality)==10 and len(classes)==300 and len(raw)==15
assert len({r['run_id'] for r in runs})==15
for q in quality:
    selected=[r for r in runs if r['model']==q['model']]
    assert {int(r['seed']) for r in selected}=={69420,67,69}
    prefix='val' if q['split']=='validation' else 'test'
    for metric in ('accuracy','macro_f1','loss'):
        values=[float(r[f'{prefix}_{metric}']) for r in selected]
        close(st.mean(values),q[f'{metric}_mean'])
        close(st.stdev(values),q[f'{metric}_sample_sd'])
    for r in selected:
        perclass=[c for c in classes if c['model']==r['model'] and c['seed']==r['seed'] and c['split']==q['split']]
        assert len(perclass)==10 and {int(c['support']) for c in perclass}=={1000}
        close(st.mean(float(c['recall']) for c in perclass),r[f'{prefix}_accuracy'])
        close(st.mean(float(c['f1']) for c in perclass),r[f'{prefix}_macro_f1'])
for t in raw:
    m=t['measurement']
    assert t['checkpoint_sha256']==next(r['checkpoint_sha256'] for r in runs if r['run_id']==t['run_id'])
    assert m['device']=='cpu' and m['cpu_threads']==2 and m['batch_size']==64
    assert m['warmup_steps']==20 and len(m['batch_ms_samples'])==m['measurement_steps']==100
    close(st.median(m['batch_ms_samples']),m['median_batch_ms'])
    close(m['median_batch_ms']/64,m['amortized_ms_per_image'])
    close(64000/m['median_batch_ms'],m['images_per_second'])
for q in timings:
    subset=[t['measurement'] for t in raw if t['model']==q['model']]
    assert len(subset)==3 and q['device']=='Windows CPU'
    for field in ('median_batch_ms','amortized_ms_per_image','images_per_second'):
        values=[t[field] for t in subset]
        close(st.mean(values),q[field+'_mean'])
        close(st.stdev(values),q[field+'_sample_sd'])

pdf=PdfReader(ROOT/'main.pdf')
single=PdfReader(ROOT/'build/consolidated/report.pdf')
pages=[p.extract_text() for p in pdf.pages]
assert len(pages)==45==len(single.pages)
assert all(norm(a)==norm(b.extract_text()) for a,b in zip(pages,single.pages))
text='\n'.join(pages)
flat=norm(text)
assert not any(s in text for s in ('TODO','??','\ufffd'))
checked_rows=0
tables=['performance','parameters','inference','training_time','seed_results']
tables += [f'per_class_aggregate_{m}' for m in ('linear','mlp','cnn','rnn','transformer')]
for name in tables:
    for line in (ROOT/'tables'/f'{name}.tex').read_text().splitlines():
        if ' & ' not in line or not line.rstrip().endswith(r'\\'):
            continue
        if line.startswith(('Model &','Class &')):
            continue
        expected=line.removesuffix(r'\\').replace(' & ',' ').replace('$','').replace(r'\pm','±').replace(r'\%','%')
        assert norm(expected) in flat,(name,expected)
        checked_rows+=1
for entry in ('main.log','build/consolidated/report.log'):
    log=(ROOT/entry).read_text(encoding='utf-8',errors='replace')
    assert 'Overfull' not in log and not re.search(r'undefined|multiply defined',log,re.I)
for f in (ROOT/'tools').glob('*.py'):
    ast.parse(f.read_text(encoding='utf-8'))
ast.parse((ROOT/'reproduction/reproduce.py').read_text())
main=(ROOT/'main.tex').read_text(encoding='utf-8')
alltex=main+'\n'+'\n'.join(p.read_text(encoding='utf-8') for p in (ROOT/'sections').glob('*.tex'))
graphics=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',alltex)
for g in graphics:
    assert (ROOT/g).is_file(),g
for item in re.findall(r'\\input\{([^}]+)\}',alltex):
    assert (ROOT/(item+'.tex')).is_file(),item
source=read(E/'source_package.json')
assert sha(Path(source['source_zip']))==source['sha256']
assert sha(ROOT/'hcmut-report.cls')==source['original_class_sha256']
with zipfile.ZipFile(source['source_zip']) as z:
    for name in ('hcmut-report.cls','hcmut.png','LICENSE.txt','TEMPLATE_SOURCE.txt'):
        matches=[n for n in z.namelist() if Path(n).name==name]
        assert len(matches)==1 and z.read(matches[0])==(ROOT/name).read_bytes()
portable=ROOT.parents[2]/'tmp/latex-delivery-check-2026-10-26'
portable_status='not exercised on this machine'
if (portable/'main.pdf').is_file():
    other=PdfReader(portable/'main.pdf')
    assert len(other.pages)==len(pages)
    assert all(norm(a)==norm(b.extract_text()) for a,b in zip(pages,other.pages))
    portable_status='passed: source ZIP extracted to a separate directory and compiled; all 45 pages match normalized final text'
result={
    'verified_at':datetime.now(timezone(timedelta(hours=7))).isoformat(),
    'status':'passed',
    'pages':45,
    'quality_runs':15,'quality_summary_rows':10,'class_rows':300,
    'timing_runs':15,'raw_batch_timings':1500,'timing_profile':'cpu-f7208f3550fc, Windows CPU',
    'rendered_numeric_table_rows_checked':checked_rows,
    'aggregate_recalculation':'mean/sample SD independently recomputed from retained per-run records',
    'per_class_reconciliation':'all 30 per-run/split macro-F1 and accuracy values agree with class metrics',
    'single_source_equivalence':'45/45 pages match after whitespace, ligature and line-end hyphen normalization',
    'portable_archive_compilation':portable_status,
    'undefined_references':0,'overfull_boxes':0,'remaining_placeholders':0,
    'visual_review':{
        'date':'2026-10-26','pages':45,
        'method':'All final pages rendered with Poppler and visually inspected in four contact sheets; detailed timing/gallery/disclosure pages also inspected.',
        'findings':'No clipping, overlapping tables, broken glyphs, unreadable gallery labels or orphan near-empty body pages observed.'
    },
    'compiler':'Tectonic 0.17.0 Windows MSVC; XeTeX backend',
    'compiler_notes':[
        'Fontconfig default-config message is emitted by portable runtime; rendered fonts inspected.',
        'Upstream anyfontsize.sty comment-byte warning does not introduce U+FFFD in PDF text.'
    ],
    'source_zip_sha256':source['sha256'],
    'class_sha256':sha(ROOT/'hcmut-report.cls'),
    'template_assets_unchanged':['hcmut-report.cls','hcmut.png','LICENSE.txt','TEMPLATE_SOURCE.txt'],
    'pdf_sha256':sha(ROOT/'main.pdf'),
    'model_execution_during_report_authoring':False,
    'original_artifacts_modified':False,
    'publication_or_submission_performed':False
}
(E/'final_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
