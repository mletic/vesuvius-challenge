"""Register my render to a published one (rotation search + phase correlation) and
report the correlation of the two ink predictions over the overlap."""
import sys
import numpy as np, tifffile, zarr
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation as pcc
from skimage.transform import rotate

pub_z, pub_ink, my_z, my_ink, pl, ml = sys.argv[1:7]
ds = 4
def ct(p, mid):
    a = zarr.open_group(p, "r")["0"]; m = int(mid)
    return a[m - 4:m + 5, ::ds, ::ds].mean(0).astype(np.float32)
A, B = ct(pub_z, pl), ct(my_z, ml)
hp = lambda x: np.where(x > 0, x - ndi.gaussian_filter(x, 6), 0)
best = None
for ang in np.arange(-40, 41, 2):
    Br = rotate(B, ang, preserve_range=True, order=1)
    s, err, _ = pcc(hp(A), hp(Br), normalization=None)
    Bs = ndi.shift(Br, s, order=1)
    m = (A > 0) & (Bs > 0)
    if m.mean() < 0.2: continue
    r = np.corrcoef(hp(A)[m], hp(Bs)[m])[0, 1]
    if best is None or r > best[0]: best = (r, ang, s)
r, ang, s = best
print(f"best CT match: rot {ang} deg, shift {s*ds} px, highpass corr {r:.3f}")
IA = tifffile.imread(pub_ink).astype(np.float32)[::ds, ::ds]
IB = ndi.shift(rotate(tifffile.imread(my_ink).astype(np.float32)[::ds, ::ds], ang, preserve_range=True, order=1), s, order=1)
m = (A > 0) & (ndi.shift(rotate(B, ang, preserve_range=True, order=1), s, order=1) > 0)
print(f"ink correlation over overlap ({m.mean():.0%}): {np.corrcoef(IA[m], IB[m])[0,1]:.3f}")
