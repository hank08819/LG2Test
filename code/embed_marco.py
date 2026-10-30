"""AI agent memory at a million: 1,000,000 passages drawn uniformly (numpy seed 0) from the 8,841,823 of the MS MARCO passage
collection, embedded with BGE-base-en-v1.5 (unit length, first 512 tokens), the encoder of the 8-K store. Saves X (float16 to halve
the file; scores use float32) and the passage ids.
usage: CUDA_VISIBLE_DEVICES=0 python embed_marco.py <collection.tsv> <out npz> [n]"""
import sys, numpy as np, torch
from sentence_transformers import SentenceTransformer
src, out = sys.argv[1:3]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000
ids, txt = [], []
with open(src, encoding="utf-8") as f:
    for line in f:
        i, t = line.rstrip("\n").split("\t", 1); ids.append(int(i)); txt.append(t)
print("collection", len(txt), flush=True)
sel = np.sort(np.random.default_rng(0).choice(len(txt), N, replace=False)); txt = [txt[i] for i in sel]; ids = np.array(ids)[sel]
st = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda"); st.max_seq_length = 512; st.half()
X = st.encode(txt, batch_size=512, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True)
np.savez(out, X=X.astype(np.float16), pid=ids)
print(X.shape, "done", flush=True)
