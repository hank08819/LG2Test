"""Pathology: CRC-VAL-HE-7K (Kather et al.; 7,180 H&E tiles, 224 x 224 px at 0.5 um/px, cut from whole-slide images of 50 colorectal
cancer patients; nine tissue classes), embedded with Phikon (owkin/phikon, ViT-B trained by masked image modeling on TCGA slides):
the class token of the last layer, scaled to unit length. Reference: cosine distance between tiles.
usage: CUDA_VISIBLE_DEVICES=0 python embed_path.py <dir holding CRC-VAL-HE-7K/> <out npz>"""
import sys, os, glob, numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
D, out = sys.argv[1:3]
files = sorted(glob.glob(f"{D}/CRC-VAL-HE-7K/*/*.tif")); lab = np.array([os.path.basename(os.path.dirname(f)) for f in files])
proc = AutoImageProcessor.from_pretrained("owkin/phikon"); model = AutoModel.from_pretrained("owkin/phikon").cuda().eval()
E = []
with torch.inference_mode():
    for i in range(0, len(files), 64):
        x = proc([Image.open(f).convert("RGB") for f in files[i:i + 64]], return_tensors="pt")["pixel_values"].cuda()
        E.append(model(pixel_values=x).last_hidden_state[:, 0, :].float().cpu().numpy())
X = np.concatenate(E); X /= np.linalg.norm(X, axis=1, keepdims=True)
np.savez_compressed(out, X=X.astype(np.float32), label=lab, fname=np.array([os.path.basename(f) for f in files]))
print(X.shape, {k: int((lab == k).sum()) for k in np.unique(lab)})
