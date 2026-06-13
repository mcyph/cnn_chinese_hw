# Architecture & Research Notes

This document explains the model used by `cnn_chinese_hw`, why it is built the
way it is, and the published research each design decision is grounded in. It
is intended both as developer documentation and as the justification for the
2024-era rewrite from a TensorFlow/Keras VGG-style model to a compact PyTorch
residual network.

## 1. Problem framing: *online* HCCR

This project does **online** handwritten Chinese character recognition (HCCR):
the input is the *trajectory* of pen/finger strokes (ordered sequences of
`(x, y)` points), not a static scanned image. This matters because the stroke
order and direction carry strong discriminative signal, and because an input
method must tolerate strokes drawn slightly out of order.

There are two dominant families of approaches in the literature:

1. **Sequence models over the raw trajectory** — e.g. the recurrent network of
   Zhang, Yin, Zhang, Liu & Bengio, *"Drawing and Recognizing Chinese
   Characters with Recurrent Neural Network"*, IEEE TPAMI 40(4):849–862, 2018
   (arXiv:1606.06539).
2. **Render-then-CNN** — convert the trajectory into a small multi-channel
   image (a *directMap*) that bakes in direction/order information, then
   classify it with a 2-D CNN.

We use approach (2). It is simpler to train and deploy than a sequence model,
and it is the representation that holds the benchmark results for online HCCR
(see §2). It is also what the original author had already hand-rolled — the
three-channel raster — so the rewrite keeps that hard-won representation and
modernises the classifier around it.

## 2. Input representation: the directMap

Each character is rendered into a `48 × 48 × 3` `uint8` image
(`recognizer/config.py`, `stroke_tools/`). The three channels encode:

| channel | what it emphasises | why |
|--------:|--------------------|-----|
| 0 | intensity ramped toward the **start** of each stroke | recovers stroke direction |
| 1 | intensity ramped toward the **end** of each stroke | recovers stroke direction |
| 2 | intensity ramped toward the **earliest strokes** | recovers stroke order |

This is a lightweight instance of the **normalisation-cooperated
direction-decomposed feature map (directMap)** that, fed into a deep CNN,
produced the state-of-the-art results in:

> Xu-Yao Zhang, Yoshua Bengio, Cheng-Lin Liu, *"Online and Offline Handwritten
> Chinese Character Recognition: A Comprehensive Study and New Benchmark"*,
> Pattern Recognition 61 (2017) 348–360. arXiv:1606.05763.

That paper showed directMap + DCNN reaches the highest accuracies on the
ICDAR-2013 online/offline competition databases *without* needing data
augmentation or model ensembling. We keep light geometric augmentation anyway
(rotation, scale, displacement, jitter, radial distortion — `HWDataAugmenter`)
because our training corpus is drawn by only a handful of writers, so synthetic
variation genuinely helps generalisation here.

The reference benchmark for this task is:

> Fei Yin, Qiu-Feng Wang, Xu-Yao Zhang, Cheng-Lin Liu, *"ICDAR 2013 Chinese
> Handwriting Recognition Competition"*, ICDAR 2013 — best reported online
> isolated-character correct rate 97.39% on CASIA-OLHWDB.

Note our training data (Tomoe + KanjiVG + supplemental) is much smaller than
CASIA, so those numbers are an aspirational ceiling, not a like-for-like target.

## 3. Backbone: SE-ResNet with global average pooling

`recognizer/model.py` defines `DirectMapSENet`:

```
stem:  3x3 conv -> BN -> GELU                          (48 ch)
stage 1: 2 x SE-residual block, stride-2 downsample    (64 ch)
stage 2: 2 x SE-residual block, stride-2 downsample    (128 ch)
stage 3: 2 x SE-residual block, stride-2 downsample    (256 ch)
head:  BN -> GELU -> global average pool -> dropout -> linear
```

Each design choice and its source:

* **Residual blocks** — He, Zhang, Ren & Sun, *"Deep Residual Learning for
  Image Recognition"*, CVPR 2016. Residual connections make the deeper,
  better-regularised stack trainable from scratch.

* **Squeeze-and-Excitation channel attention** — Hu, Shen & Sun,
  *"Squeeze-and-Excitation Networks"*, CVPR 2018. Each block recalibrates its
  channels with negligible parameter cost; consistently a free accuracy bump.

* **Global average pooling instead of a flatten + 1024/512 dense head** — the
  single biggest change. The original Keras model put two huge fully-connected
  layers after a flatten, which dominated its >30M parameters. Replacing them
  with GAP is the standard compactness technique for HCCR and removes the bulk
  of the weights with no accuracy loss:
  Xiao, Jin, Yang, Yang, Sun & Chang, *"Building Fast and Compact Convolutional
  Neural Networks for Offline Handwritten Chinese Character Recognition"*,
  arXiv:1702.07975, 2017 (a 9-layer net for 3,755 classes, compressed to 1/18
  the baseline size for a 0.21% accuracy drop).

* **GELU activations and a single norm/activation per residual function** —
  micro-design borrowed from Liu, Mao, Wu, Feichtenhofer, Darrell & Xie,
  *"A ConvNet for the 2020s"* (ConvNeXt), CVPR 2022, arXiv:2201.03545, which
  showed that modernising a plain ResNet's micro-design closes the gap to
  vision transformers while staying a pure ConvNet.

* **Global Response Normalization (GRN)** — Woo, Debnath, Hu, Chen, Liu, Kweon
  & Xie, *"ConvNeXt V2"*, CVPR 2023, arXiv:2301.00808. A lightweight per-block
  layer that normalises each channel by its global response relative to the
  others, increasing inter-channel competition and countering the feature
  collapse / redundant-activation problem of plain ConvNets. It is the identity
  at initialisation (zero-init `gamma`/`beta`), so it is on by default with
  negligible parameter cost (`ModelConfig.use_grn`).

* **Stochastic depth (DropPath)** — Huang, Sun, Liu, Sedra & Weinberger,
  *"Deep Networks with Stochastic Depth"*, ECCV 2016. Drops residual branches
  per-sample during training to regularise the stack.

* **Anti-aliased downsampling (BlurPool)** — Zhang, *"Making Convolutional
  Networks Shift-Invariant Again"*, ICML 2019 (arXiv:1904.11486). Plain strided
  convolutions violate the sampling theorem and make the network's output
  jump around under small input shifts. Each stride-2 step is therefore a
  stride-1 convolution followed by a fixed low-pass (binomial) blur and
  subsample, on both the residual and shortcut paths. Because a handwritten
  character lands at a slightly different position every time, this
  shift-robustness is directly useful here. Configurable via
  `ModelConfig.anti_alias`.

The result is **~2.9M parameters** — roughly an order of magnitude smaller than
the original dense model — which is fast to run on CPU for serving. (BlurPool
adds only fixed, non-learnable buffers, so it does not change the parameter
count.)

## 4. Training recipe

`recognizer/train.py`:

* **AdamW** (decoupled weight decay) — Loshchilov & Hutter, *"Decoupled Weight
  Decay Regularization"*, ICLR 2019.
* **Cosine LR schedule with linear warm-up** — Loshchilov & Hutter, *"SGDR:
  Stochastic Gradient Descent with Warm Restarts"*, ICLR 2017.
* **Label smoothing** (0.1) — Szegedy et al., *"Rethinking the Inception
  Architecture for Computer Vision"*, CVPR 2016; analysed in Müller, Kornblith
  & Hinton, *"When Does Label Smoothing Help?"*, NeurIPS 2019.
* **Automatic mixed precision** via `torch.amp`.
* **Exponential moving average (EMA)** of the weights for the evaluated/exported
  model — a standard, cheap accuracy/stability win. The decay is *warmed up*
  (≈0.99 early, rising to 0.999) so the averaged model does not lag near random
  initialisation during the first epochs (Tarvainen & Valpola, "Mean teachers
  are better role models", NeurIPS 2017, arXiv:1703.01780). In practice this
  matters a lot: without the warm-up the EMA model evaluated near-random for the
  first several epochs.
* **Mixup / CutMix** (optional, off by default) — convex/patch input mixing,
  the two regularisers that drive the modern "ResNet strikes back" training
  recipe (Wightman, Touvron & Jégou, arXiv:2110.00476, 2021), originally from
  Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
  (arXiv:1710.09412), and Yun et al., "CutMix", ICCV 2019 (arXiv:1905.04899).
  Enabled via `TrainConfig.mixup_alpha` / `cutmix_alpha`. They are left off by
  default because their benefit on small, fine-grained, sparse-stroke directMaps
  is recipe-dependent and best confirmed empirically on the target data — but
  the implementation is provided and verified so a full run can switch them on.
* **Class-balanced sampling** — Cui, Jia, Lin, Song & Belongie, *"Class-Balanced
  Loss Based on Effective Number of Samples"*, CVPR 2019 (arXiv:1901.05555).
  Characters present in several scripts (zh / zh-TW / ja) or with multiple
  hand-drawn copies are over-represented. We re-weight the sampler by the
  inverse *effective number of samples* per class, `(1 − βⁿ)/(1 − β)`, so rare
  characters are not drowned out. Configurable via `TrainConfig.balanced_sampling`.
* **Early stopping and checkpoint selection on top-k accuracy** (KanjiVG
  hold-out). An input method surfaces a *candidate list*, so the metric that
  matters is whether the correct character appears in the top-k, not only when
  it is ranked first — the top-k convention follows Russakovsky et al.,
  *"ImageNet Large Scale Visual Recognition Challenge"*, IJCV 2015, and matches
  the original author's own observation that the right answer "usually is in the
  top few". Both top-1 and top-k are reported each epoch.
* **Reproducibility**: Python, NumPy and torch RNGs are seeded, and each
  DataLoader worker gets a distinct deterministic seed so on-the-fly
  augmentation is varied yet repeatable.
* **Sharpness-Aware Minimization (optional)** — Foret, Kleiner, Mobahi &
  Neyshabur, *"Sharpness-Aware Minimization for Efficiently Improving
  Generalization"*, ICLR 2021 (arXiv:2010.01412), with the scale-invariant ASAM
  variant of Kwon et al., ICML 2021 (arXiv:2102.11600). SAM seeks *flat* minima
  (low loss across a neighbourhood, not just at a point), which generalise
  better. This is especially relevant here: with ~15k character classes and
  only ~1.7 training examples per class on average, the model is easy to overfit
  and flat-minima methods are most useful in exactly this small-data regime.
  Enabled with `TrainConfig.sam` (or `--sam`); it costs a second
  forward/backward per step (largely hidden because training is
  augmentation-bound) and disables AMP.

### Post-hoc confidence calibration

After training, a single **temperature** `T` is fitted on the validation set by
minimising NLL, and the logits are divided by `T` at inference (Guo, Pleiss, Sun
& Weinberger, *"On Calibration of Modern Neural Networks"*, ICML 2017,
arXiv:1706.04599). Modern networks are systematically over-confident, and an IME
*shows* the candidate scores, so calibrated probabilities are more useful than
raw softmax. Temperature scaling does not change the ranking or accuracy — only
the reported confidences. `T` is stored in the checkpoint and applied by
`HandwritingRecognizer`.

### Test-time augmentation (inference)

`HandwritingRecognizer.get_candidates_list(..., tta=N)` can average the softmax
over the clean render plus `N` randomly-augmented renders. Averaging predictions
over transformed inputs is a long-standing accuracy technique (Krizhevsky,
Sutskever & Hinton, NeurIPS 2012; Simonyan & Zisserman, ICLR 2015) and restores
the augmented-averaging the original demo performed. It defaults to off (`N=0`)
for low-latency serving.

## 5. Data pipeline

The old pipeline pre-rendered ~50 augmented rasters per character and
serialised everything to a single 1.6 GB `.npz`. The new
`recognizer/dataset.py`:

* parses the (small) stroke trajectories once and caches *those*
  (`data/stroke_cache.pkl`, tens of MB), and
* renders + augments each sample **on the fly** inside the `DataLoader`
  workers, so every epoch sees fresh augmentation and the disk/RAM footprint
  stays small.

Training data is the author's supplemental hand-drawn samples plus the Tomoe
corpus (zh / zh-TW / ja, de-duplicated across scripts). Validation is **KanjiVG**,
restricted to classes present in training and kept strictly separate — exactly
the train/validation split the original project used.

## 6. Deployment

* **In-process**: `client_server/HWServer.py` → `recognizer/recognizer.py`
  (`HandwritingRecognizer`). The recognizer is lazy-loaded so importing the
  server never requires a checkpoint to be present.
* **Portable / framework-free**: `recognizer/export_onnx.py` exports the trained
  graph to ONNX (optionally int8-quantised) for `onnxruntime`, with no
  TensorFlow dependency. This replaces the old TFLite quantisation path.

## 7. Contemporary directions & current limitations

The model above is a *closed-vocabulary classifier*: it can only output one of
the character classes it was trained on, and our training corpus (Tomoe +
supplemental, validated on KanjiVG) is small. The two limitations that current
research most directly addresses are (a) recognising characters unseen at
training time, and (b) learning from limited data. These are documented here as
the principled next steps; they are deliberately **not** implemented yet because
each is a substantial change and needs decomposition data we do not currently
ship.

### Compositional & zero-shot recognition (the main frontier)

Chinese characters are compositional — built from a few hundred *radicals*,
themselves built from ~30 *strokes*. Instead of classifying into a fixed label
set, recent work predicts the radical/stroke composition (e.g. an Ideographic
Description Sequence) and matches it to a dictionary, which lets a model
recognise characters it never saw during training:

* P. Wang, J. Zhang et al. *Radical Analysis Network (RAN): zero-shot learning
  for printed Chinese character recognition*, arXiv:1711.01889, 2017 — the
  seminal radical-sequence approach.
* J. Chen et al. *Zero-Shot Chinese Character Recognition with Stroke- and
  Radical-Level Decompositions (STAR)*, IJCNN/ICDAR 2023, arXiv:2210.08490.
* H. Zhan, Y. Li, Y.-J. Xiong, Y. Lu. *FaRE: A Feature-Aware Radical Encoding
  Strategy for Zero-Shot Chinese Character Recognition*, ACCV 2024.
* *CoLa: Chinese Character Decomposition with Compositional Latent Components*,
  arXiv:2506.03798, 2025 — a 2025 take on compositional representation.

A contained first step that fits the present codebase would be an **auxiliary
radical/stroke prediction head** (multi-task learning) alongside the existing
classifier, trained from IDS decomposition data such as CHISE/IDS or the
Tencent/`cjkvi-ids` tables. This typically improves the base classifier even
before any zero-shot decoding is added.

### Data efficiency

* X. Chen et al. *Stroke-Based Autoencoders: Self-Supervised Learners for
  Efficient Zero-Shot Chinese Character Recognition*, arXiv:2207.08191, 2022 —
  self-supervised pretraining on stroke structure, useful precisely when
  labelled data is scarce (our situation).

### Richer input encoding

Our 3-channel directMap is a lightweight hand-rolled encoding. The benchmark
representation in Zhang, Bengio & Liu (2017) is an **8-directional** decomposition
(stroke direction quantised into 8 orientation channels via the gradient/Gabor
"direction-decomposed" maps). Widening the encoder to 8 directional channels —
`CHANNELS` is already a single config knob — is the most direct path to closing
the gap with that benchmark, at a small compute cost.

## References

1. X.-Y. Zhang, Y. Bengio, C.-L. Liu. *Online and Offline Handwritten Chinese
   Character Recognition: A Comprehensive Study and New Benchmark.* Pattern
   Recognition 61 (2017) 348–360. https://arxiv.org/abs/1606.05763
2. X.-Y. Zhang, F. Yin, Y.-M. Zhang, C.-L. Liu, Y. Bengio. *Drawing and
   Recognizing Chinese Characters with Recurrent Neural Network.* IEEE TPAMI
   40(4):849–862, 2018. https://arxiv.org/abs/1606.06539
3. F. Yin, Q.-F. Wang, X.-Y. Zhang, C.-L. Liu. *ICDAR 2013 Chinese Handwriting
   Recognition Competition.* ICDAR 2013.
   https://ieeexplore.ieee.org/document/6628856/
4. X. Xiao et al. *Building Fast and Compact Convolutional Neural Networks for
   Offline Handwritten Chinese Character Recognition.* arXiv:1702.07975, 2017.
   https://arxiv.org/abs/1702.07975
5. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image
   Recognition.* CVPR 2016. https://arxiv.org/abs/1512.03385
6. J. Hu, L. Shen, G. Sun. *Squeeze-and-Excitation Networks.* CVPR 2018.
   https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html
7. Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie. *A ConvNet
   for the 2020s.* CVPR 2022. https://arxiv.org/abs/2201.03545
8. G. Huang, Y. Sun, Z. Liu, D. Sedra, K. Weinberger. *Deep Networks with
   Stochastic Depth.* ECCV 2016. https://arxiv.org/abs/1603.09382
9. I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019.
   https://arxiv.org/abs/1711.05101
10. I. Loshchilov, F. Hutter. *SGDR: Stochastic Gradient Descent with Warm
    Restarts.* ICLR 2017. https://arxiv.org/abs/1608.03983
11. C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, Z. Wojna. *Rethinking the
    Inception Architecture for Computer Vision.* CVPR 2016.
    https://arxiv.org/abs/1512.00567
12. R. Müller, S. Kornblith, G. Hinton. *When Does Label Smoothing Help?*
    NeurIPS 2019. https://arxiv.org/abs/1906.02629
13. R. Zhang. *Making Convolutional Networks Shift-Invariant Again.* ICML 2019.
    https://arxiv.org/abs/1904.11486
14. Y. Cui, M. Jia, T.-Y. Lin, Y. Song, S. Belongie. *Class-Balanced Loss Based
    on Effective Number of Samples.* CVPR 2019. https://arxiv.org/abs/1901.05555
15. O. Russakovsky et al. *ImageNet Large Scale Visual Recognition Challenge.*
    IJCV 2015. https://arxiv.org/abs/1409.0575
16. A. Krizhevsky, I. Sutskever, G. Hinton. *ImageNet Classification with Deep
    Convolutional Neural Networks.* NeurIPS 2012.
    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
17. K. Simonyan, A. Zisserman. *Very Deep Convolutional Networks for Large-Scale
    Image Recognition.* ICLR 2015. https://arxiv.org/abs/1409.1556
18. A. Tarvainen, H. Valpola. *Mean Teachers Are Better Role Models:
    Weight-Averaged Consistency Targets Improve Semi-Supervised Deep Learning
    Results.* NeurIPS 2017. https://arxiv.org/abs/1703.01780
19. R. Wightman, H. Touvron, H. Jégou. *ResNet Strikes Back: An Improved
    Training Procedure in timm.* arXiv:2110.00476, 2021.
    https://arxiv.org/abs/2110.00476
20. H. Zhang, M. Cisse, Y. Dauphin, D. Lopez-Paz. *mixup: Beyond Empirical Risk
    Minimization.* ICLR 2018. https://arxiv.org/abs/1710.09412
21. S. Yun, D. Han, S. Oh, S. Chun, J. Choe, Y. Yoo. *CutMix: Regularization
    Strategy to Train Strong Classifiers with Localizable Features.* ICCV 2019.
    https://arxiv.org/abs/1905.04899
22. P. Wang et al. *Radical Analysis Network for Zero-Shot Learning in Printed
    Chinese Character Recognition.* arXiv:1711.01889, 2017.
    https://arxiv.org/abs/1711.01889
23. J. Chen et al. *Zero-Shot Chinese Character Recognition with Stroke- and
    Radical-Level Decompositions (STAR).* 2023. https://arxiv.org/abs/2210.08490
24. H. Zhan, Y. Li, Y.-J. Xiong, Y. Lu. *FaRE: A Feature-Aware Radical Encoding
    Strategy for Zero-Shot Chinese Character Recognition.* ACCV 2024.
    https://openaccess.thecvf.com/content/ACCV2024/html/Zhan_FaRE_A_Feature-aware_Radical_Encoding_Strategy_for_Zero-shot_Chinese_Character_ACCV_2024_paper.html
25. *CoLa: Chinese Character Decomposition with Compositional Latent Components.*
    arXiv:2506.03798, 2025. https://arxiv.org/abs/2506.03798
26. X. Chen et al. *Stroke-Based Autoencoders: Self-Supervised Learners for
    Efficient Zero-Shot Chinese Character Recognition.* arXiv:2207.08191, 2022.
    https://arxiv.org/abs/2207.08191
27. S. Woo, S. Debnath, R. Hu, X. Chen, Z. Liu, I.-S. Kweon, S. Xie. *ConvNeXt
    V2: Co-designing and Scaling ConvNets with Masked Autoencoders.* CVPR 2023.
    https://arxiv.org/abs/2301.00808
28. C. Guo, G. Pleiss, Y. Sun, K. Weinberger. *On Calibration of Modern Neural
    Networks.* ICML 2017. https://arxiv.org/abs/1706.04599
29. Z.-L. Bai, Q. Huo. *A Study on the Use of 8-Directional Features for Online
    Handwritten Chinese Character Recognition.* ICDAR 2005. (Direction-decomposed
    feature maps; basis for the richer-encoding path in §7.)
30. P. Foret, A. Kleiner, H. Mobahi, B. Neyshabur. *Sharpness-Aware Minimization
    for Efficiently Improving Generalization.* ICLR 2021.
    https://arxiv.org/abs/2010.01412
31. J. Kwon, J. Kim, H. Park, I. K. Choi. *ASAM: Adaptive Sharpness-Aware
    Minimization for Scale-Invariant Learning of Deep Neural Networks.* ICML
    2021. https://arxiv.org/abs/2102.11600
