"""Export a packed dataset patch as interoperable uint8 0/1 TIFF + validity TIFF."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import tifffile
from vit_seg.binary_targets import BinaryTargetStore
from vit_seg.data import CHANNELS


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--patch-id',type=int,required=True)
    p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    target,valid,tissue=BinaryTargetStore(Path(a.dataset)/'targets').read(a.patch_id)
    for name,value in [('target',target),('valid',valid.astype(np.uint8)),('tissue',tissue.astype(np.uint8))]:
        path=out/f'patch_{a.patch_id}_{name}.tiff'
        tifffile.imwrite(path,value,photometric='minisblack',compression='zlib',metadata={'axes':'CYX' if value.ndim==3 else 'YX'})
        assert np.array_equal(tifffile.imread(path),value)
    (out/f'patch_{a.patch_id}.json').write_text(json.dumps(dict(patch_id=a.patch_id,channels=CHANNELS,
        target_values=[0,1],ignore='valid==0; do not treat ignored zeros as negatives'),indent=2))
    print(out,flush=True)


if __name__=='__main__':main()
