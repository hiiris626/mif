"""Count native-mask nearest-resized tissue, without decoding 16 target planes."""
import json,sys,zlib,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vit_seg.artifacts import file_hash
PROJECT=Path(__file__).resolve().parents[1]
FOLDER=PROJECT.parents[1]/'datasets/binary_expression_v1/targets'
OUT=PROJECT/'review';META=json.loads((FOLDER/'COMPLETE.json').read_text());INDEX=np.load(FOLDER/'index.npy',mmap_mode='r')
H,W=META['height'],META['width'];PIXELS=H*W
YS=np.floor(np.arange(256)*H/256).astype(int);XS=np.floor(np.arange(256)*W/256).astype(int)

def shard(j):
 first=j*META['shard_size'];last=min(first+META['shard_size'],META['count'])
 counts=np.zeros(last-first,np.int32);available=np.zeros((last-first,16),bool)
 with (FOLDER/f'shard_{j:05d}.bin').open('rb') as stream:
  for k,pid in enumerate(range(first,last)):
   offset,length=map(int,INDEX[pid]);stream.seek(offset)
   raw=np.frombuffer(zlib.decompress(stream.read(length)),np.uint8)
   if len(raw)!=10+2*PIXELS+(2*PIXELS+7)//8 or np.frombuffer(raw[:8],'<i8')[0]!=pid:raise ValueError('Invalid binary record')
   available[k]=np.unpackbits(raw[8:10],count=16).astype(bool)
   masks=np.unpackbits(raw[10+2*PIXELS:],count=2*PIXELS).reshape(2,H,W)
   counts[k]=masks[0][YS[:,None],XS].sum()
 return first,last,counts,available

def main():
 start=time.monotonic();counts=np.zeros(META['count'],np.int32);available=np.zeros((META['count'],16),bool)
 shards=(META['count']+META['shard_size']-1)//META['shard_size']
 with ThreadPoolExecutor(8) as pool:
  futures=[pool.submit(shard,j) for j in range(shards)]
  for n,f in enumerate(as_completed(futures),1):
   a,b,c,v=f.result();counts[a:b]=c;available[a:b]=v
   status=dict(shards_done=n,total_shards=shards,seconds=round(time.monotonic()-start,1))
   (OUT/'count_progress.json').write_text(json.dumps(status))
   if n%10==0:print(status,flush=True)
 np.savez_compressed(OUT/'all_patch_tissue_counts.npz',tissue=counts,available=available)
 (OUT/'all_patch_tissue_counts.json').write_text(json.dumps(dict(count=META['count'],grid=256,source_manifest_sha256=file_hash(FOLDER/'COMPLETE.json'),all_records_decoded=True,validity_policy='available_channel_and_tissue',seconds=time.monotonic()-start),indent=2))
 print('COUNTS COMPLETE',flush=True)
if __name__=='__main__':main()
