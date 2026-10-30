"""Finance agent memory: 5,000 SEC 8-K filings (uniform sample, numpy seed 0, of the 18,329 in sec8k_panel.parquet, read only),
embedded with BGE-base-en-v1.5 (unit length, first 512 tokens), the encoder of the companion study. Saves X, ticker, date, main item.
usage: python3 embed_8k.py <panel parquet> <out npz>"""
import sys, re, numpy as np, pyarrow.parquet as pq
from collections import Counter
from sentence_transformers import SentenceTransformer
t = pq.read_table(sys.argv[1], columns=["ticker", "date", "text"]).to_pydict(); idx = np.sort(np.random.default_rng(0).choice(len(t["text"]), 5000, replace=False))
d = {k: [v[i] for i in idx] for k, v in t.items()}
def item(t):
    for m in re.findall(r"Item\s+(\d\.\d\d)", t):
        if m != "9.01": return m
    return "none"
st = SentenceTransformer("BAAI/bge-base-en-v1.5", device="mps"); st.max_seq_length = 512
X = st.encode(d["text"], batch_size=32, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
np.savez_compressed(sys.argv[2], X=X, ticker=np.array(d["ticker"], str), date=np.array([str(x) for x in d["date"]]), item=np.array([item(x) for x in d["text"]]))
print(X.shape, Counter(item(x) for x in d["text"]).most_common(8))
