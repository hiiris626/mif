"""Binary targets (0/1) and separate validity masks, losslessly packed on disk."""
import json
import os
from pathlib import Path
import zlib
import numpy as np


class BinaryTargetStore:
    def __init__(self, folder):
        self.folder=Path(folder)
        self.meta=json.loads((self.folder/'COMPLETE.json').read_text())
        if self.meta['format']!='binary_expression_zlib_v1':
            raise ValueError('Unsupported binary target dataset')
        self.index=np.load(self.folder/'index.npy',mmap_mode='r')
        if self.index.shape!=(self.meta['count'],2):raise ValueError('Invalid binary target index')
        self.stream=None;self.open_shard=None;self.owner_pid=None

    def __getstate__(self):
        state=self.__dict__.copy();state['stream']=None;state['open_shard']=None;state['owner_pid']=None
        return state

    def read(self, patch_id):
        patch_id=int(patch_id)
        if not 0<=patch_id<self.meta['count']:raise ValueError('Binary target ID out of range')
        shard=patch_id//self.meta['shard_size']
        if self.open_shard!=shard or self.owner_pid!=os.getpid():
            if self.stream is not None:self.stream.close()
            self.stream=(self.folder/f'shard_{shard:05d}.bin').open('rb')
            self.open_shard=shard;self.owner_pid=os.getpid()
        offset,length=map(int,self.index[patch_id]);self.stream.seek(offset)
        packed=self.stream.read(length)
        if len(packed)!=length:raise ValueError('Truncated binary target')
        raw=np.frombuffer(zlib.decompress(packed),np.uint8)
        h,w=self.meta['height'],self.meta['width'];pixels=h*w
        if len(raw)!=10+2*pixels+(2*pixels+7)//8:raise ValueError('Invalid target record length')
        if int(np.frombuffer(raw[:8],'<i8')[0])!=patch_id:raise ValueError('Binary target ID mismatch')
        available=np.unpackbits(raw[8:10],count=16).astype(bool)
        target=np.unpackbits(raw[10:10+2*pixels].reshape(2,h,w),axis=0)
        tissue,eligible=np.unpackbits(raw[10+2*pixels:],count=2*pixels).reshape(2,h,w).astype(bool)
        valid=available[:,None,None]&eligible[None]
        if np.any(target[~valid]):raise ValueError('Positive target outside valid supervision')
        return target,valid,tissue
