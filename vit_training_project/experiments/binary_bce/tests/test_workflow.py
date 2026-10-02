"""Check one-command stage wiring without launching workers or accessing GPUs."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import nullcontext
from unittest.mock import patch
import pandas as pd
from vit_seg import workflow


class WorkflowTests(unittest.TestCase):
    def test_measured_gpu_headroom_and_config_binding(self):
        import torch
        cfg=json.loads(Path('configs/train.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            resolved=dict(cfg,batch_size=64,grad_accum=1)
            (folder/'resolved_config.json').write_text(json.dumps(resolved))
            (folder/'cached_gpu_verification.json').write_text(json.dumps(dict(ranks=[
                dict(rank=r,batch_size=64,real_model=True,cache_enabled=True,measured_steps=9,peak_reserved_gib=41.2)
                for r in range(4)])))
            with patch.object(torch.cuda,'device_count',return_value=4), \
                 patch.object(torch.cuda,'device',side_effect=lambda i:nullcontext()), \
                 patch.object(torch.cuda,'is_bf16_supported',return_value=True), \
                 patch.object(torch.cuda,'mem_get_info',return_value=(int(44.5*2**30),48*2**30)) as memory:
                workflow.gpu_preflight(cfg,folder)
                memory.return_value=(43*2**30,48*2**30)
                with self.assertRaisesRegex(RuntimeError,'measured workload'):
                    workflow.gpu_preflight(cfg,folder)
                memory.return_value=(int(44.5*2**30),48*2**30)
                resolved['tile_size']=512
                (folder/'resolved_config.json').write_text(json.dumps(resolved))
                with self.assertRaisesRegex(RuntimeError,'occupied'):
                    workflow.gpu_preflight(cfg,folder)

    def test_one_command_includes_validation_thresholds_before_test(self):
        original = Path.cwd()
        config = json.loads(Path('configs/train.json').read_text())
        architecture = Path('configs/virchow2_config.json').read_text()
        with tempfile.TemporaryDirectory() as temporary:
            project=Path(temporary);(project/'configs').mkdir();(project/'source').mkdir()
            (project/'weights').write_bytes(b'synthetic checkpoint placeholder')
            config.update(data_root=str(project/'source'),weights_path=str(project/'weights'))
            (project/'configs/train.json').write_text(json.dumps(config))
            (project/'configs/virchow2_config.json').write_text(architecture)
            calls=[]
            class Process:
                pid=999999
                def __init__(self,command,**kwargs):
                    calls.append(command)
                    def value(flag): return command[command.index(flag)+1]
                    if 'vit_seg.prepare' in command and 'finalize' in command:
                        (Path(value('--out'))/'COMPLETE.json').write_text('{}')
                    if 'vit_seg.capacity' in command:
                        target=Path(value('--out'));target.mkdir()
                        (target/'resolved_config.json').write_text(json.dumps(config))
                    if '--calibrate' in command:
                        assert value('--evaluate-only')=='val'
                        (Path(value('--out'))/'thresholds.json').write_text('{"fitted_split":"val"}')
                    if 'vit_seg.train_ddp' in command and '--evaluate-only' in command and value('--evaluate-only')=='test':
                        assert Path(value('--thresholds')).exists()
                def wait(self,**kwargs): return 0
                def poll(self): return 0
                def terminate(self): pass
            try:
                with patch.object(workflow,'PROJECT',project), patch.object(workflow,'gpu_preflight'), \
                     patch.object(workflow.subprocess,'Popen',Process), \
                     patch('vit_seg.prepare.source_table',return_value=pd.DataFrame({'patch_id':[0]})), \
                     patch('sys.argv',['workflow','--approved']), patch.dict(os.environ,clear=False):
                    workflow.main()
                status=json.loads((project/'runs/prepared/status.json').read_text())
                self.assertEqual(status['stage'],'complete')
                training=[c for c in calls if 'vit_seg.train_ddp' in c]
                self.assertEqual(len(training),3)
                self.assertNotIn('--evaluate-only',training[0])
                self.assertIn('--calibrate',training[1]);self.assertIn('--thresholds',training[2])
                self.assertEqual(sum('inventory' in c for c in calls),8)
                cache_index=next(i for i,c in enumerate(calls) if 'vit_seg.cache' in c)
                train_index=next(i for i,c in enumerate(calls) if 'vit_seg.train_ddp' in c)
                self.assertLess(cache_index,train_index)
            finally:
                os.chdir(original)


if __name__=='__main__': unittest.main()
