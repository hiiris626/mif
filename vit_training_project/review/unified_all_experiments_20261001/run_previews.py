"""Generate score maps for the shared qualitative patch set."""
import os,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
P=Path(__file__).resolve().parent
assign=[['v1','v5'],['v2','v6'],['v3','v9'],['v4','positive_dice_partial']]
def worker(gpu,names):
 for name in names:
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
  with (P/f'{name}_previews.log').open('w') as log:
   result=subprocess.run([sys.executable,'-B',str(P/'evaluate.py'),'--model',name,'--previews'],env=env,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(f'{name} preview inference failed')
with ThreadPoolExecutor(4) as pool:
 jobs=[pool.submit(worker,g,n) for g,n in enumerate(assign)]
 for job in jobs:job.result()
print('ALL PREVIEW SCORES COMPLETE')
