"""One fixed channel palette; argmax is visualization only, not supervision."""
import numpy as np

MARKER_COLORS = np.array([
    (.05,.28,.95),(.00,.85,.35),(.95,.10,.10),(.95,.55,.05),
    (.95,.10,.85),(.05,.85,.85),(.95,.90,.05),(.55,.10,.95),
    (.30,.75,.20),(.95,.40,.60),(.45,.95,.45),(.95,.70,.35),
    (.85,.85,.90),(.65,.35,.95),(.35,.85,.95),(.90,.20,.35),
],np.float32)
PALETTE = np.vstack((np.zeros((1,3),np.uint8),np.rint(MARKER_COLORS*255).astype(np.uint8)))


def argmax_display(values, tissue, available=None, thresholds=.5):
    """Choose the largest probability among threshold-positive channels only.

    Zero means background or no predicted positive. This never uses GT gates.
    """
    values=np.asarray(values)
    if values.ndim!=3 or values.shape[0]!=16 or values.shape[1:]!=np.shape(tissue):
        raise ValueError('Expected aligned 16-channel score image and H&E tissue mask')
    if not np.isfinite(values).all():raise ValueError('Nonfinite display scores')
    if (values<0).any() or (values>1).any():raise ValueError('Expected probabilities in [0,1], not intensities')
    threshold=np.broadcast_to(np.asarray(thresholds,float),(16,))
    if not np.isfinite(threshold).all() or ((threshold<=0)|(threshold>=1)).any():
        raise ValueError('Thresholds must be finite and strictly between zero and one')
    positive=values>=threshold[:,None,None]
    if available is not None:
        if np.shape(available)!=(16,):raise ValueError('Expected 16 channel availability flags')
        positive &= np.asarray(available,bool)[:,None,None]
    labels=(np.where(positive,values,-1).argmax(0)+1).astype(np.uint8)
    labels[~np.asarray(tissue,bool)|~positive.any(0)]=0
    return labels,PALETTE[labels]
