"""Pathology at 100,000 tiles: NCT-CRC-HE-100K (Kather et al.; 100,000 H&E tiles, 224 x 224 px at 0.5 um/px, from 86 slides; the same
nine tissue classes as CRC-VAL-HE-7K), embedded with Phikon exactly as embed_path.py (class token, unit length); images read by a
DataLoader with workers.
usage: CUDA_VISIBLE_DEVICES=0 python embed_path100k.py <dir holding NCT-CRC-HE-100K/> <out npz>"""
import sys, os, glob, numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
D, out = sys.argv[1:3]
files = sorted(glob.glob(f"{D}/NCT-CRC-HE-100K/*/*.tif")); lab = np.array([os.path.basename(os.path.dirname(f)) for f in files])
proc = AutoImageProcessor.from_pretrained("owkin/phikon"); model = AutoModel.from_pretrained("owkin/phikon").cuda().eval().half()
class DS(torch.utils.data.Dataset):
    def __len__(self): return len(files)
    def __getitem__(self, i): return proc(Image.open(files[i]).convert("RGB"), return_tensors="pt")["pixel_values"][0]
dl = torch.utils.data.DataLoader(DS(), batch_size=256, num_workers=8, pin_memory=True)
E = []
with torch.inference_mode():
    for x in dl: E.append(model(pixel_values=x.cuda(non_blocking=True).half()).last_hidden_state[:, 0, :].float().cpu().numpy())
X = np.concatenate(E); X /= np.linalg.norm(X, axis=1, keepdims=True)
np.savez(out, X=X.astype(np.float32), label=lab, fname=np.array([os.path.basename(f) for f in files]))
print(X.shape, {k: int((lab == k).sum()) for k in np.unique(lab)}, flush=True)
