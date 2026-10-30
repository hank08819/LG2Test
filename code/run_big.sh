#!/bin/bash
# MNIST-60k: prep (data, exact 10-NN), maps, then the four parameter paths, at most three runs at a time; seed 0 first.
cd ~/LHP/PAOWN/code; export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 NJOBS=8 CUDA_VISIBLE_DEVICES=""
PY=../../SC1M/env/bin/python; O=../big; mkdir -p $O ../logs
$PY exp_big.py $O prep > ../logs/big_prep.log 2>&1
$PY exp_big.py $O maps > ../logs/big_maps.log 2>&1 &
for s in 0 1 2; do
  for j in perplexity:5 perplexity:15 perplexity:30 perplexity:50 perplexity:100 perplexity:200 perplexity:500 umap:5 umap:15 umap:50 umap:100 umap:200 umap:500 exaggeration:1 exaggeration:2 exaggeration:4 exaggeration:8 exaggeration:16 exaggeration:32 pcak:2 pcak:5 pcak:10 pcak:30 pcak:50; do
    p=${j%%:*}; v=${j##*:}
    while [ $(jobs -rp | wc -l) -ge 3 ]; do sleep 20; done
    [ -f $O/sweep_${p}_${v}_s$s.json ] || $PY exp_big.py $O sweep $p $v $s > ../logs/big_${p}_${v}_s$s.log 2>&1 &
  done
done
wait; touch $O/big.done
