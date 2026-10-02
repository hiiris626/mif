"""Run only v11 after v10 completes successfully; persist queue state."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

P=Path(__file__).resolve().parents[1]
RUN=P/'runs/v11'
PREVIOUS=P.parent/'v10/runs/v10'

def read_json(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}

def save(stage,**extra):
    value=dict(stage=stage,pid=os.getpid(),time=datetime.now().astimezone().isoformat(),**extra)
    temp=RUN/'queue_status.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(RUN/'queue_status.json')

def predecessor_ready():
    status=read_json(PREVIOUS/'status.json').get('stage')
    if status=='failed_or_interrupted':raise RuntimeError('v10 failed/interrupted; v11 not started')
    with (PREVIOUS/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        if status!='complete':raise RuntimeError('v10 is not running and not complete; v11 not started')
        for relative in ['results/model/best.pt','results/model/last.pt','results/test_metrics.json']:
            if not (PREVIOUS/relative).is_file():raise RuntimeError('v10 completion artifact missing: '+relative)
    return True

def main():
    if '--approved' not in sys.argv:raise SystemExit('Explicit training authorization required')
    if not read_json(RUN/'prelaunch_check.json').get('passed'):raise RuntimeError('v11 prelaunch checks missing')
    with (RUN/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            save('waiting_for_v10')
            while not predecessor_ready():time.sleep(30)
            save('starting_v11')
            with (RUN/'launcher.log').open('a') as log:
                child=subprocess.Popen(['bash','scripts/run.sh','--approved','--resume','--config','configs/v11.json','--run-dir','runs/v11'],cwd=P,stdout=log,stderr=subprocess.STDOUT)
                save('running_v11',child_pid=child.pid)
                code=child.wait()
            if code:raise RuntimeError(f'v11 workflow exited {code}; inspect launcher.log')
            save('complete')
        except BaseException as exc:
            save('failed_or_interrupted',error=str(exc));raise

if __name__=='__main__':main()
