"""Check a published segment as a micro-seg control: map a surface-volume (row, col)
to level-2 coords via its tifxyz mesh and show the surface-prediction profile along
the mesh normal there (a peak at offset 0 means mesh and prediction agree)."""
import io, sys, requests, tifffile, numpy as np
from scipy import ndimage as ndi
from vesuvius_sweep.microseg import SOURCES, open_arr
src, mesh, H, W, row, col = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:7])
B = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
X = {a: tifffile.imread(io.BytesIO(requests.get(f"{B}{mesh}/{a}.tif").content)).astype(float) for a in "zyx"}
h, w = X["z"].shape
i, j = int(row * h / H), int(col * w / W)
win = lambda A: A[max(0, i - 15):i + 15, max(0, j - 15):j + 15]
P = np.stack([win(X[a]).ravel() for a in "zyx"], -1); P = P[(P > 0).all(1)]
P2 = (P - 1.5) / 4
c = np.array([X[a][i, j] for a in "zyx"]); c2 = (c - 1.5) / 4
tr = np.array([np.nanmedian(np.gradient(win(X[a]), axis=0)) for a in "zyx"])
tc = np.array([np.nanmedian(np.gradient(win(X[a]), axis=1)) for a in "zyx"])
n = np.cross(tr, tc); n /= np.linalg.norm(n)
print("L2 centre zyx:", c2.round().astype(int).tolist(), " mesh normal:", n.round(3), " rows-dir:", (tr / np.linalg.norm(tr)).round(2), " cols-dir:", (tc / np.linalg.norm(tc)).round(2))
sf_a = open_arr(SOURCES[src][1], "0")
lo = np.floor(c2 - 60).astype(int); sf = sf_a[tuple(slice(a, a + 120) for a in lo)].astype(np.float32)
prof = [ndi.map_coordinates(sf, (P2 - lo + n * k).T, order=1).mean() for k in range(-12, 13)]
print("surface-pred along mesh normal, offsets -12..12:", np.round(prof).astype(int).tolist())
