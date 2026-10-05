# Abliteration overlay for the 2.75 bpw TensorFold champion

Live weights: `/mnt/spark2-models-local/GLM-5.3-EXL3-2.75-mixedK-EXL3NE-ablit`
(hardlink overlay of `…/GLM-5.3-EXL3-2.75-mixedK-EXL3NE`; stock tree is untouched).

Donor: [drowzeys/keys-GLM-5.3-EXL3-Abliterated](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-Abliterated)
(Blackfrost rank-1 derisk, α=3.0, baked into native F16 `o_proj` / residual `down_proj` for layers 2–49).

The 2.75 bpw TensorFold checkpoint stores those linears as EXL3 K5 `mul1`, so the F16 tensors
are re-quantized to the same K5 layout and spliced into shards `model-00002` and `model-00003`.
Routed experts, layers 0–1, layers 50–78 (including MTP), embeddings and the head stay stock.

```sh
# encode K5 packs (fork150 image, two ranks in parallel)
#   .1: --layers 2-25    .3: --layers 26-49
python3 ablit/encode_ablit_k5.py --out /work/k5 --layers 2-25

# splice on the NFS-exporting node (.2), never write through the stock hardlinks
python3 ablit/splice_overlay.py \
  --stock /home/keyspark/models-local/GLM-5.3-EXL3-2.75-mixedK-EXL3NE \
  --dest  /home/keyspark/models-local/GLM-5.3-EXL3-2.75-mixedK-EXL3NE-ablit \
  --k5    /home/keyspark/glm53-tf-ablit-work/k5
```

Rollback: `MODEL=/mnt/spark2-models-local/GLM-5.3-EXL3-2.75-mixedK-EXL3NE ./one-shot.sh up`
