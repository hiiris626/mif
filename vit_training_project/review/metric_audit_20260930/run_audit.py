import os,subprocess,sys,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path(__file__).resolve().parent

def worker(gpu,names):
 for name in names:
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
  with (root/f'{name}.log').open('w') as log:
   result=subprocess.run([sys.executable,str(root/'evaluate.py'),'--model',name],env=env,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(f'{name} failed: see log')
with ThreadPoolExecutor(4) as pool:
 jobs=[pool.submit(worker,i,names) for i,names in enumerate([['v1','baseline'],['v2'],['v3'],['binary_bce']])]
 for j in jobs:j.result()
print('ALL FIVE MODEL AUDITS COMPLETE',flush=True)
