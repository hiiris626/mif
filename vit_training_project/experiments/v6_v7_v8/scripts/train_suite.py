"""Sequential authorized four-GPU suite. Stop the queue on any failed stage."""
import argparse,subprocess,sys,json,os,signal,fcntl
from pathlib import Path
from datetime import datetime
P=Path(__file__).resolve().parents[1]

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--approved',action='store_true');args=parser.parse_args()
 if not args.approved:raise SystemExit('Pass --approved only after explicit training authorization')
 checks=json.loads((P/'review/CHECKS_PASSED.json').read_text());assert checks['all_passed']
 def interrupted(signum,frame):raise InterruptedError(f'Queue signal {signum}')
 signal.signal(signal.SIGTERM,interrupted)
 with (P/'runs/queue.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=dict(order=['v6','v7','v8'],completed=[],current=None,stage='starting',pid=os.getpid())
  def save():
   state['time']=datetime.now().astimezone().isoformat();temp=P/'runs/queue_status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(P/'runs/queue_status.json')
  for name in state['order']:
   state.update(current=name,stage='running');save();print('START',name,flush=True)
   with (P/'runs'/name/'launcher.log').open('a') as log:
    child=subprocess.Popen(['bash','scripts/run.sh','--approved','--resume','--config',f'configs/{name}.json','--run-dir',f'runs/{name}'],cwd=P,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
     code=child.wait()
     if code:raise RuntimeError(f'{name} workflow exited {code}; queue stopped')
    except BaseException as error:
     if child.poll() is None:
      os.killpg(child.pid,signal.SIGTERM)
      try:child.wait(timeout=45)
      except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     state.update(stage='failed_or_interrupted',error=str(error));save();raise
   state['completed'].append(name);save();print('COMPLETE',name,flush=True)
  state.update(current=None,stage='complete');save()
if __name__=='__main__':main()
