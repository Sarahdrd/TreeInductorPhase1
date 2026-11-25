"""Inventory prospective Git content without staging, removing or publishing."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/repository_review'


def git_paths(*args):
    return set(subprocess.check_output(['git',*args,'-z'],cwd=ROOT).decode().split('\0'))-{''}


def run():
    tracked=git_paths('ls-files')
    untracked=git_paths('ls-files','--others','--exclude-standard')
    paths=sorted(p for p in tracked|untracked if (ROOT/p).is_file())
    rows=[];absolute=[];secrets=[]
    patterns=[r'AKIA[0-9A-Z]{16}',r'gh[pousr]_[A-Za-z0-9]{30,}',
              r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
              r'sk-[A-Za-z0-9_-]{40,}']
    for name in paths:
        p=ROOT/name;raw=p.read_bytes()
        rows.append(dict(path=name,git_state='tracked' if name in tracked else 'untracked',
                         bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        if p.suffix in {'.py','.json','.md','.txt','.toml','.ipynb','.yaml','.yml'}:
            text=raw.decode('utf8',errors='replace')
            for i,line in enumerate(text.splitlines(),1):
                if re.search(r'[A-Za-z]:[\\/]+(?:Users|home)',line):absolute.append(dict(path=name,line=i))
                if any(re.search(pattern,line) for pattern in patterns):secrets.append(dict(path=name,line=i))
    # Exclude the inventory's own changing digest; list its final bytes through
    # git diff/status rather than claiming a recursive self-checksum.
    manifest_rows=[r for r in rows if r['path'] not in {
        'experiments/repository_review/prospective_files.tsv',
        'experiments/repository_review/inventory_summary.json',
        'experiments/repository_review/stage-paths.txt'}]
    (OUT/'prospective_files.tsv').write_text('path\tgit_state\tbytes\tsha256\n'+
        '\n'.join('\t'.join(str(r[k]) for k in ['path','git_state','bytes','sha256']) for r in manifest_rows)+'\n',encoding='utf8')
    stage=sorted(set(paths)|{'experiments/repository_review/prospective_files.tsv',
                            'experiments/repository_review/inventory_summary.json',
                            'experiments/repository_review/stage-paths.txt'})
    (OUT/'stage-paths.txt').write_text('\n'.join(stage)+'\n',encoding='utf8')
    summary=dict(prospective_file_count=len(stage),tracked=len(tracked),
                 untracked=len(untracked),prospective_bytes=sum(r['bytes'] for r in rows),
                 files_over_10_MiB=[r for r in rows if r['bytes']>10*1024**2],
                 largest_files=sorted(rows,key=lambda r:r['bytes'],reverse=True)[:10],
                 credential_pattern_findings=secrets,machine_specific_provenance_locations=absolute,
                 note='Pattern scan is not a guarantee of absence of secrets. Immutable historical provenance may retain recorded executable/source paths. No git mutation performed.')
    (OUT/'inventory_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:summary[k] for k in ['prospective_file_count','tracked','untracked','prospective_bytes','files_over_10_MiB','credential_pattern_findings']},indent=2))
    print('Recorded absolute-path locations:',len(absolute))


if __name__=='__main__':run()
