# Local and global fidelity of embeddings, and the LG reversal test

Code and results for the paper *Quantifying Local and Global Fidelity of Nonlinear Embeddings*.
The locality profile reads a local fidelity $L$ (k = 10) and a global fidelity $G$ (mean over 0.3 <= k/n <= 0.5)
from one chance-corrected statistic. The LG reversal test decides whether one map is more local and another
more global, with a block-jackknife p-value and an interval for the crossover scale.

## Contents

| Folder | What it holds |
|---|---|
| `lgfid/` | Python package: `lg_scores`, `reversal_test`, profile plot, command line `lgfid score` / `lgfid test` |
| `srtsne2/` | SR-tSNE2, the rank refinement of t-SNE used in the paper (Han, Niu, Li and Wang, *Differential rank preservation manifold learning*, Pattern Recognition, under review) |
| `code/` | Data preparation, experiments, figures and tables of the paper |
| `data/` | Every result file behind the figures and tables, packed in five containers |
| `unpack_data.py` | Restores the result files to `results/` and `v6_inputs/` with their original names |

## Quick start

```bash
pip install -r requirements.txt
pip install -e lgfid
python3 lgfid/examples/quickstart.py          # L, G and the reversal test on the scikit-learn digits
python3 unpack_data.py .                       # restores results/ and v6_inputs/
```

## License

Code: MIT License (see `LICENSE`). The result files in `data/` are derived from public datasets and follow their terms.
