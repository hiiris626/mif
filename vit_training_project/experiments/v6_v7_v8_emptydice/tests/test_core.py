import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
from vit_seg.data import CHANNELS, IGNORE, augment, labels_from_mif, valid_flag
from vit_seg.distributed import EvaluationSampler, DistributedPixelLoss, lr_factor, resolve_batch
from vit_seg.prepare import inventory, finalize, patient_split, balanced_view
from datacore.training_monitor import EarlyStopper
from vit_seg.validate import validate


class CoreTests(unittest.TestCase):
    def test_patient_split(self):
        df = pd.DataFrame(dict(orion_slide_id=np.repeat([f"P{i}" for i in range(41)], 2), split="old"))
        result = patient_split(df)
        self.assertEqual(result.groupby("split").orion_slide_id.nunique().to_dict(), dict(train=29,val=6,test=6))
        self.assertTrue((result.groupby("orion_slide_id").split.nunique() == 1).all())

    def test_balancing_and_unavailable(self):
        df = pd.DataFrame(dict(patch_id=range(9), X_valid=[True]*8+[False],
                               X_pixels=[0]*4+[1]*4+[0], X_coverage=[0]*4+[.1]*2+[.9]*2+[0]))
        view, counts = balanced_view(df, "X", .5, 42)
        self.assertEqual(view.stratum.value_counts().to_dict(), dict(false=4,true1=2,true2=2))
        self.assertEqual(counts["unavailable"], 1)
        self.assertEqual(counts["untrusted_zero"], 0)

    def test_missing_dapi_coexpression_and_background(self):
        mif = np.zeros((16,16,16), np.uint8); mif[1:3,4:12,4:12] = 10
        target, _ = labels_from_mif(mif, np.ones((16,16),bool), [10]*16, "multilabel")
        self.assertTrue((target[1:3,8,8] == 1).all())
        self.assertEqual(target[0,8,8], 0)
        self.assertEqual(target[0,0,0], IGNORE)
        self.assertFalse(valid_flag("False")); self.assertFalse(valid_flag(np.nan))

    def test_augmentation_labels_and_mask(self):
        he = np.full((16,16,3),120,np.uint8)
        target = np.full((16,16,16),IGNORE,np.uint8); target[:,4:12,4:12]=0; target[1:3,4:12,4:12]=1
        mask = target[0] != IGNORE
        for seed in range(12):
            image, labels, tissue = augment(he.copy(), target.copy(), mask.copy(), np.random.default_rng(seed))
            self.assertTrue((image[~tissue] == 0).all())
            self.assertTrue((labels[:,~tissue] == IGNORE).all())
            self.assertTrue(np.array_equal(labels[1], labels[2]))

    def test_loss_no_background_gradient(self):
        for mode in ("dice","lovasz_hinge"):
            logits = torch.randn(2,3,8,8,requires_grad=True)
            target = torch.full_like(logits, IGNORE, dtype=torch.long)
            target[:,:2,2:6,2:6]=1
            loss = DistributedPixelLoss([1,1,100], mode)(logits,target)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(logits.grad[target == IGNORE].abs().sum().item(),0)

    def test_nonpadding_validation(self):
        for n in (1,3,11,16):
            samples = [i for rank in range(4) for i in EvaluationSampler(range(n),rank,4)]
            self.assertEqual(sorted(samples),list(range(n)))

    def test_schedule_batch_and_patience(self):
        self.assertAlmostEqual(lr_factor(1,100,10,.05),.1)
        self.assertEqual(lr_factor(10,100,10,.05),1)
        self.assertAlmostEqual(lr_factor(100,100,10,.05),.05)
        resolved = resolve_batch(dict(world_size=4,target_global_batch=256),40)
        self.assertEqual(resolved,dict(batch_size=40,grad_accum=2,effective_global_batch=320))
        stopper=EarlyStopper(patience=8,min_delta=.001,warmup=10,mode="max")
        stops=[stopper.update(.4)[1] for _ in range(18)]
        self.assertFalse(any(stops[:17])); self.assertTrue(stops[17])

    def test_inventory_without_nuclear_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); rows=[]
            for i,split in enumerate(("train","val","test")):
                cv2.imwrite(str(root/f"{i}.png"),np.full((16,16,3),120,np.uint8))
                mif=np.full((16,16,16),10 if i == 0 else 200,np.uint8); mif[...,0]=0
                tifffile.imwrite(root/f"{i}.tiff",mif)
                rows.append(dict(patch_id=i,orion_slide_id=f"p{i}",in_slide_name=f"s{i}",
                    split=split,original_split=split,image_path=f"{i}.png",target_path=f"{i}.tiff",nuclei_path="unused"))
            args=argparse.Namespace(root=str(root),out=str(root/"data"),split_policy="patient_70_15_15",seed=42,
                                    limit=0,shards=1,shard=0,batch_size=3,io_workers=1)
            with patch("vit_seg.prepare.source_table",return_value=pd.DataFrame(rows)):
                inventory(args); finalize(args)
            stats=json.loads((root/"data/statistics.json").read_text())
            self.assertEqual(stats["n_retained"],3)
            self.assertEqual(stats["positive_pixel_mean"][1],10)
            self.assertFalse(stats["nuclear_qc_performed"])
            self.assertNotIn("tensorflow", __import__('sys').modules)

    def test_full_prepare_validate_and_missing_view_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); metadata=[]
            for i,split in enumerate(("train","val","test")):
                rows=[]
                for j in range(4):
                    name=f"{i}_{j}"
                    cv2.imwrite(str(root/f"{name}.png"),np.full((16,16,3),120,np.uint8))
                    mif=np.zeros((16,16,16),np.uint8)
                    mif[...,2]=20
                    if j >= 2: mif[:(2 if j == 2 else 12),:,1:]=10+j
                    tifffile.imwrite(root/f"{name}.tiff",mif)
                    rows.append(dict(in_slide_name=f"s{i}",image_path=f"{name}.png",target_path=f"{name}.tiff",nuclei_path="unused"))
                pd.DataFrame(rows).to_csv(root/f"{split}_dataframe.csv",index=False)
                metadata.append(dict(in_slide_name=f"s{i}",orion_slide_id=f"p{i}"))
            pd.DataFrame(metadata).to_csv(root/"slide_dataframe.csv",index=False)
            args=argparse.Namespace(root=str(root),out=str(root/"data"),split_policy="original",seed=42,
                                    limit=0,shards=1,shard=0,batch_size=4,io_workers=1)
            inventory(args); finalize(args)
            receipt=validate(root/"data")
            self.assertEqual(receipt["n_retained"],12)
            self.assertEqual(receipt["patient_counts"],dict(train=1,val=1,test=1))
            path=root/"data/train_balanced.csv"; frame=pd.read_csv(path)
            frame[frame.balance_channel != 'CD31'].to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,"omitted"):
                validate(root/"data")


if __name__ == "__main__": unittest.main()
