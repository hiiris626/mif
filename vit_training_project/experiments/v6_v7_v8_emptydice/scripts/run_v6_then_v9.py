"""Resume original v6, then run revised-loss v9; never launch v7/v8."""
import argparse,json,os,signal,subprocess,fcntl
from pathlib import Path
from datetime import datetime
P=Path(__file__).resolve().parents[1]
OLD=P.with_name('v6_v7_v8')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--approved',action='store_true');a=parser.parse_args()
 if not a.approved:raise SystemExit('Explicit training authorization required')
 assert json.loads((P/'review/CHECKS_PASSED.json').read_text())['all_passed']
 def interrupt(signum,frame):raise InterruptedError(f'Signal {signum}')
 signal.signal(signal.SIGTERM,interrupt)
 with (P/'runs/v6_then_v9.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=dict(order=['original_v6','v9'],completed=[],disabled=['v7','v8'],pid=os.getpid())
  def save():
   state['time']=datetime.now().astimezone().isoformat();tmp=P/'runs/v6_then_v9_status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(P/'runs/v6_then_v9_status.json')
  for name,root,version,stage in [('original_v6',OLD,'v6','train'),('v9',P,'v9','all')]:
   state.update(current=name,stage='running');save();print('START',name,flush=True)
   with (root/'runs'/version/'launcher.log').open('a') as log:
    child=subprocess.Popen(['bash','scripts/run.sh','--approved','--resume','--stage',stage,'--config',f'configs/{version}.json','--run-dir',f'runs/{version}'],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
     code=child.wait()
     if code:raise RuntimeError(f'{name} exited {code}')
    except BaseException as error:
     if child.poll() is None:
      os.killpg(child.pid,signal.SIGTERM)
      try:child.wait(timeout=45)
      except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     state.update(stage='failed_or_interrupted',error=str(error));save();raise
   state['completed'].append(name);save()
  state.update(current=None,stage='complete');save()
if __name__=='__main__':main()
