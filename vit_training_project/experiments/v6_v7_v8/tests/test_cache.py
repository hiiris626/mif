import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import DataLoader
from vit_seg.cache import build, NativeCache
from vit_seg.data import CHANNELS, PixelDataset
from vit_seg.train_ddp import initialize_worker


class CacheTests(unittest.TestCase):
    def test_lossless_cache_augmentation_epoch_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); data=root/'data'; data.mkdir()
            rng=np.random.default_rng(13); rows=[]
            for i in range(4):
                # Odd native dimensions exercise mask bit padding.
                he=rng.integers(50,200,(33,33,3),dtype=np.uint8); he[:5]=255
                mif=rng.integers(0,15,(33,33,17),dtype=np.uint8)
                mif[10:20,10:20]=0; mif[...,0]=0
                cv2.imwrite(str(root/f'{i}.png'),he)
                tifffile.imwrite(root/f'{i}.tiff',mif)
                row=dict(patch_id=i,image_path=f'{i}.png',target_path=f'{i}.tiff')
                row.update({f'{c}_valid':not(i==1 and n in (0,2)) for n,c in enumerate(CHANNELS)})
                if i==3: row.update({f'{c}_valid':False for c in CHANNELS})
                rows.append(row)
            frame=pd.DataFrame(rows)
            frame.to_csv(data/'patch_manifest.csv',index=False)
            frame.to_csv(data/'train.csv',index=False)
            stats=dict(q=[10]*16,source_spec=dict(root=str(root)))
            (data/'statistics.json').write_text(json.dumps(stats))
            cache=root/'cache'; build(data,cache,workers=2,reserve_gib=0)
            build(data,cache,workers=1,reserve_gib=0)
            plain=PixelDataset(data/'train.csv',stats,root,32,True,seed=42)
            cached=PixelDataset(data/'train.csv',stats,root,32,True,seed=42,cache_dir=cache)
            for epoch in range(8):
                plain.epoch=cached.epoch=epoch
                for i in range(4):
                    a,b=plain[i],cached[i]
                    for key in ('image','label','tissue'):
                        self.assertTrue(torch.equal(a[key],b[key]),(epoch,i,key))
                    self.assertEqual(b['label'].dtype,torch.uint8)
            loader=DataLoader(cached,batch_size=2,num_workers=1,persistent_workers=True,
                              worker_init_fn=initialize_worker)
            cached.epoch=9; first=next(iter(loader))['image'].clone()
            cached.epoch=10; second=next(iter(loader))['image'].clone()
            self.assertFalse(torch.equal(first,second))
            self.assertTrue(torch.equal(second[0],cached[0]['image']))
            del loader
            # Incomplete caches fail closed; completed caches detect damaged records.
            done=cache/'COMPLETE.json'; saved=done.read_text(); done.unlink()
            with self.assertRaises(FileNotFoundError): NativeCache(cache,data)
            done.write_text(saved)
            reader=NativeCache(cache,data)
            with (cache/'records.raw').open('r+b') as stream:
                stream.seek(100); old=stream.read(1); stream.seek(100)
                stream.write(bytes([old[0]^1]))
            with self.assertRaisesRegex(ValueError,'Corrupt'): reader.read(0)
            (data/'statistics.json').write_text(json.dumps(dict(stats,changed=True)))
            with self.assertRaisesRegex(ValueError,'source'): NativeCache(cache,data)


if __name__=='__main__': unittest.main()
