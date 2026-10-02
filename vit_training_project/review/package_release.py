"""Create a bounded source/report snapshot for the user's GitHub repository."""
import os,shutil,json,hashlib,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEST=Path('/tmp/mif_publish_20261002')
assert (DEST/'.git').is_dir()
extensions={'.py','.sh','.json','.yaml','.yml','.toml','.ini','.cfg','.md','.txt','.rst'}
excluded={'runs','results','datasets','cache','weights','__pycache__','.git','.codex','.agents','logs','node_modules','.venv'}
manifest=[]
def copy(src,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    manifest.append({'path':str(target.relative_to(DEST)),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
for folder in ['vit_training_project','vit_versions','vit_matte','datacore','configs','eval']:
    for base,dirs,files in os.walk(ROOT/folder,followlinks=True):
        dirs[:]=[d for d in dirs if d not in excluded and not d.startswith('.')]
        for name in files:
            src=Path(base)/name
            if src.suffix not in extensions or not src.is_file() or src.stat().st_size>5_000_000:continue
            if name in ['package_release.py']:continue
            copy(src,DEST/src.relative_to(ROOT))
# Preserve frozen run configurations, fitted weights, small metrics and histories.
P=ROOT/'vit_training_project'
paths={'v4':P/'runs/ddp_baseline','positive_dice_partial':P/'runs/ddp_positive_dice',
       'v5':P/'experiments/binary_bce/runs/prepared','v6':P/'experiments/v6_v7_v8/runs/v6',
       'v7':P/'experiments/v6_v7_v8/runs/v7','v8':P/'experiments/v6_v7_v8/runs/v8',
       'v9':P/'experiments/v6_v7_v8_emptydice/runs/v9','v10':P/'experiments/v10/runs/v10','v11':P/'experiments/v11/runs/v11'}
for name,run in paths.items():
    for relative in ['status.json','submitted_config.json','provenance.json','environment.json','capacity/resolved_config.json',
        'data/bce_pixel_weights.json','data/binary_targets.json','data/training_policy.json','data/statistics.json',
        'results/test_metrics.json','results/test_metrics_default05.json','results/thresholds.json',
        'results/model/config.json','results/model/classification_history.json','results/model/lora_audit.json',
        'results/model/train_log.csv','sampling_audit.csv','results/patch_balance.csv','prelaunch_check.json']:
        src=run/relative
        if src.is_file():copy(src,DEST/'experiment_records'/name/relative)
report=ROOT/'reports/experiment_report_20261002'
for src in report.rglob('*'):
    if src.is_file() and src.suffix in {'.pdf','.csv','.json','.xlsx','.png','.log'}:copy(src,DEST/'reports/experiment_report_20261002'/src.relative_to(report))
(DEST/'release_manifest.json').write_text(json.dumps(manifest,indent=2))
# Compile without producing bytecode and reject obvious secret-bearing files.
errors=[]
secret=re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}')
for item in manifest:
    f=DEST/item['path']
    if f.suffix in extensions:
        text=f.read_text(errors='replace')
        if secret.search(text):errors.append('possible secret: '+item['path'])
        if f.suffix=='.py':
            try:compile(text,str(f),'exec')
            except SyntaxError as exc:errors.append(str(exc))
if errors:raise RuntimeError('\n'.join(errors))
print('PACKAGED',len(manifest),'files',sum(x['bytes'] for x in manifest),'bytes; syntax/secret checks passed')
