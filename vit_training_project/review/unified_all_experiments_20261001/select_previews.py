"""Choose diverse, deterministic CRC02 patches for qualitative comparison."""
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent
data=pd.read_csv(P.parents[1]/'experiments/binary_bce/runs/prepared/data/test.csv')
data=data[data.patient_id=='CRC02'].copy()
markers=['CD31','CD45','CD68','CD4','FOXP3','CD8a','CD45RO','CD20','PDL1','CD3e','CD163','ECadherin','Ki67','Pan-CK','SMA']
for c in markers:data[c+'_frac']=data[c+'_pixels']/data.tissue_pixels.clip(lower=1)
data['immune']=data[[c+'_frac' for c in ['CD45','CD68','CD4','FOXP3','CD8a','CD45RO','CD20','CD3e','CD163']]].sum(axis=1)
data['epithelial']=data[['ECadherin_frac','Pan-CK_frac']].sum(axis=1)
data['total_marker']=data[[c+'_frac' for c in markers]].sum(axis=1)
data['expressed_channels']=(data[[c+'_pixels' for c in markers]]>0).sum(axis=1)
specs=[('near_negative','total_marker',True),('low_signal','total_marker',True),('many_channels','expressed_channels',False),
       ('immune_dense','immune',False),('epithelial_dense','epithelial',False),('CD68_dense','CD68_frac',False),
       ('CD163_dense','CD163_frac',False),('CD31_dense','CD31_frac',False),('CD4_dense','CD4_frac',False),
       ('FOXP3_dense','FOXP3_frac',False),('PDL1_dense','PDL1_frac',False),('SMA_dense','SMA_frac',False)]
chosen=[];used=set()
for label,column,ascending in specs:
 candidates=data.sort_values([column,'patch_id'],ascending=[ascending,True])
 # The second low-signal example comes from the lower quartile rather than
 # duplicating the absolute minimum.
 if label=='low_signal':candidates=candidates.iloc[len(candidates)//5:]
 row=next(r for _,r in candidates.iterrows() if int(r.patch_id) not in used)
 used.add(int(row.patch_id));chosen.append(dict(selection=label,patch_id=int(row.patch_id),patient_id=row.patient_id,
   image_path=row.image_path,target_path=row.target_path,tissue_pixels=int(row.tissue_pixels),
   total_marker_fraction=float(row.total_marker),expressed_channels=int(row.expressed_channels),
   immune_fraction=float(row.immune),epithelial_fraction=float(row.epithelial),selected_feature=float(row[column])))
out=pd.DataFrame(chosen);out.to_csv(P/'preview_manifest.csv',index=False)
print(out.to_string(index=False))
