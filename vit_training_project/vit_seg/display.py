"""One fixed channel palette; argmax is visualization only, not supervision."""
import numpy as np

MARKER_COLORS = np.array([
    (.05,.28,.95),(.00,.85,.35),(.95,.10,.10),(.95,.55,.05),
    (.95,.10,.85),(.05,.85,.85),(.95,.90,.05),(.55,.10,.95),
    (.30,.75,.20),(.95,.40,.60),(.45,.95,.45),(.95,.70,.35),
    (.85,.85,.90),(.65,.35,.95),(.35,.85,.95),(.90,.20,.35),
],np.float32)
PALETTE = np.vstack((np.zeros((1,3),np.uint8),np.rint(MARKER_COLORS*255).astype(np.uint8)))


def argmax_display(values, tissue, available=None):
    values=np.asarray(values)
    if values.ndim!=3 or values.shape[0]!=16 or values.shape[1:]!=np.shape(tissue):
        raise ValueError('Expected aligned 16-channel score image and H&E tissue mask')
    if not np.isfinite(values).all():raise ValueError('Nonfinite display scores')
    if available is not None:
        values=np.where(np.asarray(available,bool)[:,None,None],values,0)
    labels=(values.argmax(0)+1).astype(np.uint8)
    labels[~np.asarray(tissue,bool)|(values.max(0)<=0)]=0
    return labels,PALETTE[labels]
