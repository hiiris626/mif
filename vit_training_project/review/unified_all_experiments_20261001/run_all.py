"""Run matched inference across four GPUs without modifying checkpoints."""
import os,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent
ASSIGNMENTS=[['v1','v5'],['v2','v6'],['v3','v9'],['v4','positive_dice_partial']]
def worker(gpu,names):
 for name in names:
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
  with (ROOT/f'{name}.log').open('w') as log:
   result=subprocess.run([sys.executable,'-B',str(ROOT/'evaluate.py'),'--model',name],env=env,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(f'{name} failed; inspect {ROOT/name}.log')
with ThreadPoolExecutor(4) as pool:
 jobs=[pool.submit(worker,gpu,names) for gpu,names in enumerate(ASSIGNMENTS)]
 for job in jobs:job.result()
print('ALL MATCHED INFERENCE COMPLETE',flush=True)
