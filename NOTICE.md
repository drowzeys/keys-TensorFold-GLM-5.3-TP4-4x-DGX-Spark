# Notice

keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark — the recipe (scripts, benches, documentation): Copyright 2026 drowzeys,
MIT License (`LICENSE`). The engine and the image are separate works with their own licenses, listed below.

## The engine (in the prebuilt image, not in this repository)

- **TensorFold** — https://github.com/ashhart/TensorFold — Copyright 2026 TensorFold contributors (Ash Hart). Apache
  License 2.0 from v0.6.0; code written before v0.6.0 keeps its MIT notice. The image carries TensorFold with the
  `glm_moe_dsa` family (branch https://github.com/drowzeys/TensorFold/tree/glm53-tp4-opt), including its `NOTICE`,
  `LICENSES/` and `THIRD_PARTY_NOTICES.md`, which list every third-party source below in detail.
- **MiaAI-Lab, GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold** —
  https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold — Copyright 2026 MiaAI-Lab, Apache License
  2.0. Adapted from its patches: prompt experts (glm-prompt-kernels / -experts-order / exl3-prompt-experts), 0002
  (FP8 head), 0008 / 0015 / 0042 / 0063 (prompt reuse), 0013 (late token stream, pinned candidates), 0016 (fused expert
  epilogues), 0034 (nucleus union), 0043 (visible-key idea), 0046 (L2 prefetch: kernel used unchanged), 0047 (16-byte
  trellis loads), 0065 (rank checks), 0070 (stop vote), the dual-rail setup.
- **Jay Leaton, glm53-tensorfold-spark** — https://github.com/jayleaton/glm53-tensorfold-spark — Copyright 2026 Jay
  Leaton, Apache License 2.0. MiaAI-Lab's 0046 and 0047 are adapted from its patches 0460 and 0580.
- **BertholomusAI (Albert Lee)** — https://github.com/bertholomus/TensorFold/tree/glm-dsa-tp4 and
  https://github.com/bertholomus/glm-5.3-tensorfold-tp4-4xgb10 — Copyright 2026 Albert Lee, Apache License 2.0. Ideas
  re-implemented (no code copied): decode side stream, MTP index reuse, draft depth policy, draft cut for concurrent
  rounds, warm-up graph capture, quick fills. `bench/h2h.py` uses the four prompts of its `bench/tf_greedy.py`.
- **cuda-exl3** — the stacked expert layout and grouped prompt GEMM (a library in the image).
- **Anthropic's Claude (Claude Code)** — engineering assistance, recorded in the commits' co-author lines.
- **b12x** — https://github.com/local-inference-lab/b12x — the b12x contributors, Apache License 2.0 (RoCE one-shot
  collectives, a library in the image).
- **ExLlamaV3** — https://github.com/turboderp-org/exllamav3 — Copyright (c) 2025 Turboderp, MIT (the EXL3 format).
- **NVIDIA PyTorch container and NCCL** — the image's base; NVIDIA's license terms apply to the NVIDIA software in it.

## Models (not in this repository)

- **GLM-5.3** — Z.ai (zai-org). The quantized weights (drowzeys/keys-GLM-5.3-EXL3-2.75BPW) stay under the GLM-5.3
  license. The abliterated pack is drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated (Blackfrost derisk from
  drowzeys/keys-GLM-5.3-EXL3-Abliterated, re-quantized to EXL3 K5).
- **DFlash2 drafter (optional)** — incoai/GLM-5.3-DFlash2, CC BY-NC-ND 4.0 (non-commercial, no derivatives). Users
  download it themselves; it is never redistributed here, and the default configuration does not use it.
