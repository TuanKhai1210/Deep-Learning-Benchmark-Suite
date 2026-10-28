"""Expand local TeX inputs into the equivalent single-source report.tex."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
def expand(text,seen=()):
    def sub(match):
        path=ROOT/(match.group(1)+('.tex' if not match.group(1).endswith('.tex') else ''))
        path=path.resolve()
        if not path.is_relative_to(ROOT) or path in seen:
            raise ValueError(f'Invalid input dependency: {path}')
        return expand(path.read_text(encoding='utf-8'),seen+(path,))
    return re.sub(r'\\input\{([^}]+)\}',sub,text)

if __name__=='__main__':
    source=expand((ROOT/'main.tex').read_text(encoding='utf-8'))
    (ROOT/'report.tex').write_text('% Consolidated equivalent of main.tex; edit either source form.\n'+source,encoding='utf-8')
    print('Created report.tex; external class/logo/figures remain required.')
