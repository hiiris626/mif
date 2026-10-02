"""Complete the common CRC02 evaluation for v7/v10/v11."""
import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('matched_evaluation',P/'evaluate.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
for name,relative in [('v7','v6_v7_v8/runs/v7'),('v10','v10/runs/v10'),('v11','v11/runs/v11')]:
    run=e.PROJECT/'experiments'/relative/'results'
    e.MODELS[name]=('classification',run/'model/best.pt',run/'thresholds.json')
if __name__=='__main__':e.main()
