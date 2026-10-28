"""Create an editable report archive with only current sources and retained evidence."""
from pathlib import Path
import hashlib,json,shutil,zipfile
from datetime import datetime,timezone,timedelta

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
shutil.copyfile(ROOT/'main.pdf',ROOT/'Tiki-taka_A1_Report.pdf')
files=[ROOT/p for p in ['main.tex','report.tex','hcmut-report.cls','hcmut.png',
                       'LICENSE.txt','TEMPLATE_SOURCE.txt','README.md','Tiki-taka_A1_Report.pdf']]
for dirname in ('sections','tables','figures','evidence','reproduction','tools'):
    for p in (ROOT/dirname).rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        if p.name in ('compiled_text.txt','compiled_text_initial.txt','file_manifest.json'):
            continue
        if p.suffix in ('.tex','.png','.json','.csv','.md','.py','.ipynb','.txt'):
            files.append(p)
files=sorted(set(files))
manifest=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
manifestpath=ROOT/'evidence/file_manifest.json'
manifestpath.write_text(json.dumps({'files':manifest,'note':'Manifest excludes itself to avoid a recursive hash.'},indent=2)+'\n',encoding='utf-8')
files.append(manifestpath)
archive=ROOT.parent/'A1_Final_Completed_2026-10-26.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in files:
        z.write(p,p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(files)==len(set(z.namelist()))
    for item in manifest:
        assert hashlib.sha256(z.read(item['path'])).hexdigest()==item['sha256']
    for name in z.namelist():
        assert not name.startswith(('/','\\')) and '..' not in Path(name).parts
record={'verified_at':datetime.now(timezone(timedelta(hours=7))).isoformat(),
        'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
        'entries':len(files),'crc_check':'passed','manifest_hash_checks':len(manifest),
        'pdf':str(ROOT/'Tiki-taka_A1_Report.pdf'),'pdf_sha256':sha(ROOT/'Tiki-taka_A1_Report.pdf')}
(ROOT.parent/'completed_delivery_validation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
