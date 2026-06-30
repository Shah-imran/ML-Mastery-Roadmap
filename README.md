# ML Mastery Roadmap

A self-study plan for going from math foundations to building a large language model from scratch. Covers three textbooks, 160+ StatQuest videos, and 38 foundational papers — ordered so every step assumes only what came before it.

**Books**
- 📐 *Mathematics for Machine Learning* — Deisenroth, Faisal, Ong (MML)
- 🔥 *Deep Learning* — Goodfellow, Bengio, Courville (DL)
- 🤖 *Build a Large Language Model From Scratch* — Raschka (LLM)

**Videos** — StatQuest with Josh Starmer (160+ videos, woven throughout all phases)

**Papers** — 38 foundational papers, 2012–2023

---

## Table of Contents

- [Overview](#overview)
- [Phase 1 — Math Foundations](#phase-1--math-foundations)
- [Phase 2 — Classical Machine Learning](#phase-2--classical-machine-learning)
- [Phase 3 — Deep Learning](#phase-3--deep-learning)
- [Phase 3b — Generative Models](#phase-3b--generative-models)
- [Phase 3c — Self-Supervised Learning](#phase-3c--self-supervised-learning)
- [Phase 4 — Build a Large Language Model](#phase-4--build-a-large-language-model)
- [Papers — Full Reading List](#papers--full-reading-list)

---

## Overview

| Phase | Focus | Primary resources |
|---|---|---|
| Phase 1 | Math foundations | MML Ch. 1–7, Goodfellow Ch. 2–4, StatQuest Stats Fundamentals |
| Phase 2 | Classical ML | Goodfellow Ch. 5, StatQuest ML playlist, MML Ch. 8–12 |
| Phase 3 | Deep learning | Goodfellow Ch. 6–20, StatQuest Neural Networks & Transformers |
| Phase 3b | Generative models | VAE → GAN → DCGAN → VQ-VAE → StyleGAN → CLIP → DDPM → LDM |
| Phase 3c | Self-supervised learning | MoCo → SimCLR → BYOL → DINO → Barlow Twins → MAE |
| Phase 4 | Build an LLM | Raschka Ch. 1–7, scaling and alignment papers |

No timelines. Work through each step in order — each one builds on the last.

---

## Phase 1 — Math Foundations

Build the mathematical language that every ML concept is written in. This phase is the slowest — budget extra time here.

### Step 1 — Read AlexNet cold (orientation)

> 📄 **Paper 1:** [ImageNet Classification with Deep CNNs (AlexNet)](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) — Krizhevsky, Sutskever, Hinton (NeurIPS 2012)

Before opening any textbook, read this 9-page paper. You won't understand everything. That's the point. It shows you where the field started and what your entire study plan is building toward.

### Step 2 — Linear algebra

> 📘 **MML Chapter 2** — Linear Algebra
> 📘 **MML Chapter 3** — Analytic Geometry

- Vectors, matrices, systems of linear equations, vector spaces, linear independence, basis, rank
- Norms, inner products, projections, rotations, orthogonal complements
- Do every exercise. These chapters are the grammar of everything.

**StatQuest alongside:** Statistics Fundamentals playlist — normal distribution, covariance, histograms, standard deviation vs standard error.

### Step 3 — Matrix decompositions

> 📘 **MML Chapter 4** — Matrix Decompositions

Determinants, eigenvalues/eigenvectors, eigendecomposition, SVD, PCA. SVD is the most important decomposition in ML — spend extra time on it.

**StatQuest alongside:** PCA step-by-step, PCA in Python, LDA clearly explained, MDS and PCoA.

### Step 4 — Vector calculus

> 📘 **MML Chapter 5** — Vector Calculus

Gradients, partial derivatives, the chain rule, Taylor series, Jacobians, Hessians. The chain rule section is critical — backpropagation is just the chain rule applied recursively.

### Step 5 — Probability and information theory

> 📘 **MML Chapter 6** — Probability and Distributions
> 📘 **Goodfellow Chapter 3** — Probability and Information Theory

Read together. Goodfellow Ch. 3 adds entropy, KL divergence, and cross-entropy — the foundation of every loss function you will ever use.

**StatQuest alongside:** Probability is not Likelihood, Maximum Likelihood clearly explained, The Binomial Distribution, p-values.

> **Milestone:** Be able to derive MLE from scratch and explain why minimising cross-entropy equals maximising log-likelihood.

### Step 6 — Continuous optimization

> 📘 **MML Chapter 7** — Continuous Optimization
> 📘 **Goodfellow Chapter 4** — Numerical Computation

Gradient descent, convexity, Lagrange multipliers, constrained optimization, numerical stability.

**StatQuest alongside:** Gradient Descent Step-by-Step, Stochastic Gradient Descent Clearly Explained.

### Step 7 — ML basics framing

> 📘 **Goodfellow Chapter 5** — Machine Learning Basics (first pass)

Read as an overview: the learning paradigm, capacity, overfitting, bias-variance, regularization, No Free Lunch.

**StatQuest alongside:** Bias and Variance, Cross Validation.

### Step 8 — Dropout paper

> 📄 **Paper 2:** [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) — Srivastava, Hinton et al. (JMLR 2014)

Now that you understand probability and optimization, read this properly. Dropout appears in every neural network you will ever train.

---

## Phase 2 — Classical Machine Learning

### Step 9 — Word embeddings

> 📄 **Paper 3:** [Efficient Estimation of Word Representations in Vector Space (Word2Vec)](https://arxiv.org/abs/1301.3781) — Mikolov et al. (ICLR 2013)

Conceptual ancestor of token embeddings in LLMs. Read before going deeper into classifiers.

**StatQuest alongside:** Word Embedding and Word2Vec, Clearly Explained.

### Step 10 — Classification fundamentals

> 📘 **Goodfellow Chapter 5** — deep dive

Maximum likelihood as a framework for supervised learning, logistic regression, SVMs, k-NN, decision trees.

**StatQuest alongside:** A Gentle Introduction to ML, Logistic Regression (all 3 parts), Confusion Matrix, Sensitivity and Specificity, ROC and AUC, K-nearest neighbors, Naive Bayes.

### Step 11 — Decision trees and ensembles

Watch StatQuest in this exact order:

1. Decision Trees (3-part series)
2. Regression Trees, Clearly Explained
3. How to Prune Regression Trees
4. Random Forests Part 1 — Building, Using and Evaluating
5. Random Forests Part 2 — Missing data and clustering
6. AdaBoost, Clearly Explained
7. Gradient Boost (4-part series)
8. XGBoost (4-part series)

### Step 12 — Regularization

**StatQuest:** Ridge (L2), Lasso (L1), Ridge vs Lasso Visualized, Elastic Net, One-Hot/Label/Target Encoding.

### Step 13 — Clustering and dimensionality reduction

**StatQuest:** K-means clustering, Hierarchical Clustering, t-SNE Clearly Explained, PCA in Python, PCA Practical Tips.

### Step 14 — Support vector machines

**StatQuest:** SVMs Part 1 (Main Ideas), Part 2 (Polynomial Kernel), Part 3 (RBF Kernel).

Connect to **MML Chapter 12** (Classification) for the rigorous mathematical treatment.

### Step 15 — Entropy and information-theoretic losses

**StatQuest:** Entropy, Clearly Explained; Mutual Information, Clearly Explained.

### Step 16 — Adam optimizer

> 📄 **Paper 4 (reading order) / Paper 5 (overall):** [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Kingma & Ba (ICLR 2015)

The default optimizer for virtually every modern neural network.

### Step 17 — GANs: first look

> 📄 **Paper 5 (reading order) / Paper 4 (overall):** [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — Goodfellow et al. (NeurIPS 2014)

9-page paper. Read as a taste of generative modeling before Phase 3.

### Step 18 — MML application chapters

> 📘 **MML Chapters 9–12** — selectively

- Ch. 9: Linear Regression (probabilistic treatment)
- Ch. 10: Dimensionality Reduction with PCA
- Ch. 11: Density Estimation with Gaussian Mixture Models
- Ch. 12: Classification (SVM deep dive)

> **Milestone:** Implement logistic regression, a decision tree, and a random forest from scratch in Python before moving on.

---

## Phase 3 — Deep Learning

### Step 19 — Neural networks and backpropagation

> 📘 **Goodfellow Chapter 6** — Deep Feedforward Networks

Watch StatQuest first, then read:

- Neural Networks Pt. 1: Inside the Black Box
- Neural Networks Pt. 2: Backpropagation Main Ideas
- Neural Networks Pt. 3: ReLU in Action
- Neural Networks Pt. 4: Multiple Inputs and Outputs

> **Mandatory exercise:** Implement backpropagation in NumPy — no PyTorch autograd — for at least one small two-layer network.

### Step 20 — Batch normalization

> 📄 **Paper 7:** [Batch Normalization: Accelerating Deep Network Training](https://arxiv.org/abs/1502.03167) — Ioffe & Szegedy (ICML 2015)

Made training very deep networks stable and fast. Note: transformers use Layer Norm instead — this paper is why you learn the distinction.

### Step 21 — Regularization for deep learning

> 📘 **Goodfellow Chapter 7** — Regularization for Deep Learning

L1/L2, dropout, data augmentation, early stopping, batch normalization as regularizer.

### Step 22 — Optimization for deep models

> 📘 **Goodfellow Chapter 8** — Optimization for Training Deep Models

SGD, momentum, Adam in context, learning rate schedules, second-order methods. The most practically important chapter in the book.

### Step 23 — VGGNet

> 📄 **Paper 6:** [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGGNet)](https://arxiv.org/abs/1409.1556) — Simonyan & Zisserman (ICLR 2015)

First systematic study of depth. Key insight: many small (3×3) filters beat a few large ones.

### Step 24 — Convolutional neural networks

> 📘 **Goodfellow Chapter 9** — Convolutional Networks

**StatQuest alongside:** Neural Networks CNN series (full sequence).

> Switch to PyTorch for all implementations from this point. Raschka (Phase 4) assumes PyTorch fluency.

### Step 24a — U-Net

> 📄 **Paper 8:** [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger, Fischer, Brox (MICCAI 2015)

Maps every pixel to a label using encoder-decoder + skip connections. The denoising U-Net inside Stable Diffusion is a direct descendant of this architecture.

**Key ideas:**
- Contracting path: capture context, reduce spatial resolution
- Expanding path: restore spatial resolution for precise localisation
- Skip connections: concatenate encoder features to decoder at each level

### Step 25 — ResNet

> 📄 **Paper 9:** [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385) — He, Zhang, Ren, Sun (CVPR 2016, Best Paper)

Skip connections make arbitrarily deep networks trainable. This idea appears directly in transformers as the residual stream around each attention and FFN sublayer.

### Step 26 — Sequence modeling and RNNs

> 📘 **Goodfellow Chapter 10** — Sequence Modeling: Recurrent and Recursive Nets

**StatQuest alongside:** RNN series, LSTM series.

### Step 27 — Seq2Seq

> 📄 **Paper 10:** [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — Sutskever, Vinyals, Le (NeurIPS 2014)

Encoder LSTM compresses an input sequence to a fixed vector; decoder LSTM generates output. Direct predecessor of the transformer.

### Step 28 — Bahdanau Attention

> 📄 **Paper 11:** [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — Bahdanau, Cho, Bengio (ICLR 2015)

**Read before "Attention Is All You Need."** Introduced soft attention: the decoder looks back at all encoder hidden states with learned weights. The Q/K/V formulation in transformers is a direct generalization.

### Step 29 — The Transformer

> 📄 **Paper 12:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani, Shazeer, Parmar et al. (NeurIPS 2017)

The most important paper in modern AI. Eliminated recurrence entirely. Every modern LLM descends from this 15-page paper.

**StatQuest first:** Attention and Transformer series (full sequence).

**Key ideas:**
- Multi-head self-attention: each head learns a different type of relationship
- Scaled dot-product: Q·Kᵀ / √d_k prevents softmax saturation
- Positional encodings inject order without recurrence
- Encoder-decoder with cross-attention; GPT uses only the decoder half

### Step 30 — BERT

> 📄 **Paper 13:** [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — Devlin, Chang, Lee, Toutanova (NAACL 2019)

Masked Language Modeling pretraining that transfers to almost any NLP task. Contrasting BERT (encoder-only, bidirectional) with GPT (decoder-only, autoregressive) is essential.

### Step 31 — GPT-2

> 📄 **Paper 14:** [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al. (OpenAI 2019)

The model you build in Raschka's book is GPT-2. Read this before Phase 4.

### Step 32 — Vision Transformer (ViT)

> 📄 **Paper 17:** [An Image is Worth 16×16 Words (ViT)](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al. (ICLR 2021)

Proved the transformer architecture works for vision by treating image patches as tokens. No convolutions at all.

### Step 32a — Swin Transformer

> 📄 **Paper 18:** [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Ze Liu et al. (ICCV 2021, Best Paper)

Fixed ViT's O(n²) limitation with hierarchical patch merging and shifted window attention. Replaced CNNs as the backbone for detection and segmentation.

**The architectural synthesis:** U-Net (multi-scale encoder-decoder) + ResNet (skip connections) + ViT (transformer attention) = Swin.

### Step 33 — Practical methodology and applications

> 📘 **Goodfellow Chapter 11** — Practical Methodology
> 📘 **Goodfellow Chapter 12** — Applications

Hyperparameter search, debugging strategies, end-to-end system design.

### Step 34 — DDPM

> 📄 **Paper 31:** [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) — Ho, Jain, Abbeel (NeurIPS 2020)

Learns to reverse a gradual Gaussian noising process. The denoising network is a U-Net. Surpassed GANs on image quality; eliminated mode collapse.

### Step 34a — Latent Diffusion Models (Stable Diffusion)

> 📄 **Paper 32:** [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Rombach, Blattmann et al. (CVPR 2022 Oral)

Runs DDPM in a VAE's compressed latent space. The text conditioning uses CLIP embeddings injected via cross-attention at each U-Net layer. Stable Diffusion is a synthesis of VAE + U-Net + Transformer cross-attention + CLIP + DDPM.

### Step 35 — Goodfellow Part III (selective)

> 📘 **Goodfellow Chapters 14, 20** — read in full
> 📘 **Goodfellow Chapters 13, 15–19** — read as reference

---

## Phase 3b — Generative Models

Read alongside Goodfellow Ch. 14 and Ch. 20. Each paper solves a problem left open by the previous.

### Step G1 — VAE

> 📄 **Paper 25:** [Auto-Encoding Variational Bayes (VAE)](https://arxiv.org/abs/1312.6114) — Kingma & Welling (ICLR 2014)

Learns a structured continuous latent space via the reparameterisation trick and the ELBO. Conceptual ancestor of every latent-space generative model, including the encoder inside Stable Diffusion.

**Key ideas:**
- Encoder outputs μ and σ of a Gaussian; reparameterisation trick makes sampling differentiable
- ELBO = reconstruction loss + KL divergence (pulls latent toward standard Gaussian)
- Smooth latent space: interpolating between z vectors gives semantically smooth outputs

### Step G2 — DCGAN

> 📄 **Paper 26:** [Unsupervised Representation Learning with Deep Convolutional GANs (DCGAN)](https://arxiv.org/abs/1511.06434) — Radford, Metz, Chintala (ICLR 2016)

Made GANs work in practice with convolutional architectures, batch normalisation, and leaky ReLU. Launched a decade of GAN research.

### Step G3 — VQ-VAE

> 📄 **Paper 27:** [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937) — van den Oord, Vinyals, Kavukcuoglu (NeurIPS 2017)

Replaces the continuous Gaussian latent with a discrete learned codebook. No posterior collapse. Opens the door to autoregressive modelling of image tokens. Seeds the idea behind DALL-E's tokenisation.

### Step G4 — Progressive GAN

> 📄 **Paper 28:** [Progressive Growing of GANs](https://arxiv.org/abs/1710.10196) — Karras, Aila, Laine, Lehtinen (ICLR 2018)

Grows generator and discriminator together from 4×4 to 1024×1024. First GAN to produce convincing megapixel photorealistic faces.

### Step G5 — StyleGAN

> 📄 **Paper 29:** [A Style-Based Generator Architecture for GANs (StyleGAN)](https://arxiv.org/abs/1812.04948) — Karras, Laine, Aila (CVPR 2019)

Mapping network z→w + per-layer AdaIN style injection disentangles high-level attributes from fine detail. The per-layer conditioning idea is the same mechanism used in Stable Diffusion's U-Net cross-attention.

### Step G6 — CLIP

> 📄 **Paper 30:** [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — Radford, Kim, Hallacy et al. (ICML 2021)

Aligns images and text in a shared embedding space using contrastive training on 400M internet pairs. Zero-shot: without seeing ImageNet during training, matches a fully supervised ResNet-50. The CLIP text encoder is the conditioning pathway in Stable Diffusion.

---

## Phase 3c — Self-Supervised Learning

Read after ViT (Step 32) since most SSL methods use ViT backbones.

### Step SSL1 — MoCo

> 📄 **Paper 19:** [Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)](https://arxiv.org/abs/1911.05722) — He, Fan, Wu, Xie, Girshick (CVPR 2020)

Large-scale contrastive learning via a queue of stored negatives + a momentum encoder (slow EMA update) to keep them consistent. Decouples number of negatives from batch size.

### Step SSL2 — SimCLR

> 📄 **Paper 20:** [A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)](https://arxiv.org/abs/2002.05709) — Chen, Kornblith, Norouzi, Hinton (ICML 2020)

Large batch (4096–8192) where every other image is a negative. Key findings: augmentation composition matters most; a nonlinear projection head dramatically improves representations and is discarded after pretraining.

### Step SSL3 — BYOL

> 📄 **Paper 21:** [Bootstrap Your Own Latent (BYOL)](https://arxiv.org/abs/2006.07733) — Grill, Strub, Altché et al. (NeurIPS 2020)

No negative pairs at all. Online network predicts target network's representation of a differently augmented view. Stop-gradient on the target + momentum update prevents collapse without any negatives.

### Step SSL4 — DINO

> 📄 **Paper 22:** [Emerging Properties in Self-Supervised Vision Transformers (DINO)](https://arxiv.org/abs/2104.14294) — Caron, Touvron, Misra et al. (ICCV 2021)

Self-distillation on ViTs reveals that attention heads spontaneously segment objects with no segmentation labels — a property that doesn't appear with supervised training. k-NN on DINO features reaches 78.3% ImageNet without fine-tuning.

### Step SSL5 — Barlow Twins

> 📄 **Paper 23:** [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230) — Zbontar, Jing, Misra, LeCun, Deny (ICML 2021)

One objective: make the cross-correlation matrix between two views' embeddings equal to the identity matrix. Diagonal = 1 (each dimension is informative); off-diagonal = 0 (no redundancy). No negatives, no momentum encoder, no large batch requirement.

### Step SSL6 — MAE

> 📄 **Paper 24:** [Masked Autoencoders Are Scalable Vision Learners (MAE)](https://arxiv.org/abs/2111.06377) — He, Chen, Xie et al. (CVPR 2022)

Mask 75% of image patches; train encoder on the visible 25% only; lightweight decoder reconstructs masked pixels. Asymmetric design (encoder sees no mask tokens) gives 3× training speedup. ViT-Huge with MAE reaches 87.8% on ImageNet — the visual analogue of BERT's MLM.

---

## Phase 4 — Build a Large Language Model

Raschka's book is your guide. You build a working GPT-2 sized LLM end-to-end in PyTorch, then fine-tune it two ways.

### Step 36 — GPT-3 (read before opening the book)

> 📄 **Paper 15:** [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al. (NeurIPS 2020)

175B parameters. Few-shot in-context learning with no gradient updates. Established that scale alone is a viable research direction.

### Step 37 — Scaling Laws

> 📄 **Paper 16:** [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al. (OpenAI 2020)

Loss follows clean power laws with model size, tokens, and compute. Later revised by Chinchilla.

### Step 38 — Raschka Chapter 1: Understanding LLMs

High-level architecture of GPT-style models. Cross-reference GPT-2 and GPT-3 papers.

### Step 39 — Raschka Chapter 2: Working with Text Data

Tokenization from scratch, BPE, token embeddings, positional encodings, data loader.

> Run every notebook from the book's GitHub repo. Don't just read the code — execute it, then modify it.

### Step 40 — Raschka Chapter 3: Coding Attention Mechanisms

Self-attention, causal (masked) attention, multi-head attention — all implemented in PyTorch before any high-level abstraction.

**Rewatch:** StatQuest Attention series alongside this chapter.

### Step 41 — FlashAttention

> 📄 **Paper 33:** [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao, Fu, Ermon, Rudra (NeurIPS 2022)

Attention is O(n²) in memory. FlashAttention fixes this by reordering computation to use GPU SRAM instead of HBM for attention intermediates. 2–4× speedup; makes 16K+ token sequences practical.

### Step 42 — Raschka Chapter 4: Implementing a GPT Model from Scratch

Layer norm, feed-forward layers, residual connections, full transformer block, text generation.

> **Milestone:** ~300 lines of PyTorch. Generates garbage text — the architecture is correct.

### Step 43 — Raschka Chapter 5: Pretraining + Appendix D

Cross-entropy and perplexity evaluation, training loop, decoding strategies (greedy, top-k, top-p), loading pretrained GPT-2 weights from OpenAI.

Appendix D: learning rate warmup, cosine decay, gradient clipping.

> **Milestone:** Your architecture loads real GPT-2 weights and generates coherent English.

### Step 44 — Chinchilla

> 📄 **Paper 34:** [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) — Hoffmann et al. (DeepMind 2022)

GPT-3 and most LLMs of the era were undertrained. For a given compute budget, scale model parameters and training tokens equally. Chinchilla (70B, 1.4T tokens) outperforms Gopher (280B, 300B tokens).

### Step 45 — Raschka Chapter 6: Fine-Tuning for Classification

Add a classification head, fine-tune on spam detection, evaluate.

### Step 46 — InstructGPT

> 📄 **Paper 35:** [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — Ouyang et al. (OpenAI 2022)

The paper behind ChatGPT. Three-stage RLHF: (1) SFT on human demonstrations, (2) reward model from human preference comparisons, (3) PPO to maximise the reward model score. A 1.3B InstructGPT is preferred over 175B GPT-3 — size is not alignment.

### Step 47 — Raschka Chapter 7: Fine-Tuning to Follow Instructions

Instruction fine-tuning, instruction datasets, instruct model training and evaluation.

### Step 48 — LoRA + Raschka Appendix E

> 📄 **Paper 36:** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al. (Microsoft, ICLR 2022)

ΔW = BA where B and A are low-rank matrices. 10,000× fewer trainable parameters, comparable performance. Merged into W at inference — no added latency. How virtually all production fine-tuning is done today.

### Step 49 — Chain-of-Thought

> 📄 **Paper 37:** [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — Wei et al. (Google 2022)

"Let's think step by step" dramatically improves multi-step reasoning. Emergent above ~100B parameters. Intermediate tokens give the model additional computation per problem.

### Step 50 — Llama (final paper)

> 📄 **Paper 38:** [Llama: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al. (Meta AI 2023)

Chinchilla-optimal training + RMSNorm + SwiGLU + RoPE + open weights. Llama-13B outperforms GPT-3 (175B) on most benchmarks. Enabled the Alpaca, Vicuna, and Mistral ecosystems.

> **Roadmap complete.** You have worked through three major textbooks, 160+ StatQuest videos, and 38 foundational papers. You have built a working GPT-2 sized LLM from scratch, pretrained it, loaded real weights, and fine-tuned it two ways.

---

## Papers — Full Reading List

38 papers in the order you should read them. Each section assumes only the papers before it.

### Section 1 — Orientation

| # | Paper | Authors | Year | Category | Link |
|---|---|---|---|---|---|
| 1 | ImageNet Classification with Deep CNNs **(AlexNet)** | Krizhevsky, Sutskever, Hinton | 2012 | CNN | [PDF](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) |
| 2 | Dropout: A Simple Way to Prevent Neural Networks from Overfitting | Srivastava, Hinton et al. | 2014 | Training | [JMLR](https://jmlr.org/papers/v15/srivastava14a.html) |

### Section 2 — Classical ML and representations

| # | Paper | Authors | Year | Category | Link |
|---|---|---|---|---|---|
| 3 | Efficient Estimation of Word Representations in Vector Space **(Word2Vec)** | Mikolov et al. | 2013 | NLP | [arXiv:1301.3781](https://arxiv.org/abs/1301.3781) |
| 4 | Generative Adversarial Nets **(GAN)** | Goodfellow et al. | 2014 | Generative | [arXiv:1406.2661](https://arxiv.org/abs/1406.2661) |
| 5 | Adam: A Method for Stochastic Optimization | Kingma & Ba | 2014 | Training | [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) |

### Section 3 — CNN architectures

| # | Paper | Authors | Year | Category | Link |
|---|---|---|---|---|---|
| 6 | Very Deep Convolutional Networks for Large-Scale Image Recognition **(VGGNet)** | Simonyan & Zisserman | 2014 | CNN | [arXiv:1409.1556](https://arxiv.org/abs/1409.1556) |
| 7 | Batch Normalization: Accelerating Deep Network Training | Ioffe & Szegedy | 2015 | Training | [arXiv:1502.03167](https://arxiv.org/abs/1502.03167) |
| 8 | U-Net: Convolutional Networks for Biomedical Image Segmentation | Ronneberger, Fischer, Brox | 2015 | CNN / Segmentation | [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) |
| 9 | Deep Residual Learning for Image Recognition **(ResNet)** | He, Zhang, Ren, Sun | 2015 | CNN | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |

### Section 4 — Sequence models and attention

| # | Paper | Authors | Year | Category | Link |
|---|---|---|---|---|---|
| 10 | Sequence to Sequence Learning with Neural Networks **(Seq2Seq)** | Sutskever, Vinyals, Le | 2014 | RNN / NLP | [arXiv:1409.3215](https://arxiv.org/abs/1409.3215) |
| 11 | Neural Machine Translation by Jointly Learning to Align and Translate **(Bahdanau Attention)** | Bahdanau, Cho, Bengio | 2015 | Attention | [arXiv:1409.0473](https://arxiv.org/abs/1409.0473) |
| 12 | Attention Is All You Need **(Transformer)** | Vaswani, Shazeer, Parmar et al. | 2017 | Transformer | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |

### Section 5 — Language pretraining

| # | Paper | Authors | Year | Category | Link |
|---|---|---|---|---|---|
| 13 | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Devlin, Chang, Lee, Toutanova | 2018 | LLM | [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) |
| 14 | Language Models are Unsupervised Multitask Learners **(GPT-2)** | Radford et al. | 2019 | LLM | [OpenAI PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| 15 | Language Models are Few-Shot Learners **(GPT-3)** | Brown et al. | 2020 | LLM | [arXiv:2005.14165](https://arxiv.org/abs/2005.14165) |
| 16 | Scaling Laws for Neural Language Models | Kaplan et al. | 2020 | LLM / Scaling | [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) |

### Section 6 — Vision transformers

| # | Paper | Authors | Year | Category | Link |
|---|---|---|---|---|---|
| 17 | An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale **(ViT)** | Dosovitskiy et al. | 2020 | Vision Transformer | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) |
| 18 | Swin Transformer: Hierarchical Vision Transformer using Shifted Windows | Ze Liu, Lin, Cao et al. | 2021 | Vision Transformer | [arXiv:2103.14030](https://arxiv.org/abs/2103.14030) |

### Section 7 — Self-supervised learning

| # | Paper | Authors | Year | Category | Link |
|---|---|---|---|---|---|
| 19 | Momentum Contrast for Unsupervised Visual Representation Learning **(MoCo)** | He, Fan, Wu, Xie, Girshick | 2019 | SSL / Contrastive | [arXiv:1911.05722](https://arxiv.org/abs/1911.05722) |
| 20 | A Simple Framework for Contrastive Learning of Visual Representations **(SimCLR)** | Chen, Kornblith, Norouzi, Hinton | 2020 | SSL / Contrastive | [arXiv:2002.05709](https://arxiv.org/abs/2002.05709) |
| 21 | Bootstrap Your Own Latent **(BYOL)** | Grill, Strub, Altché et al. | 2020 | SSL / Self-distillation | [arXiv:2006.07733](https://arxiv.org/abs/2006.07733) |
| 22 | Emerging Properties in Self-Supervised Vision Transformers **(DINO)** | Caron, Touvron, Misra et al. | 2021 | SSL / Self-distillation | [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) |
| 23 | Barlow Twins: Self-Supervised Learning via Redundancy Reduction | Zbontar, Jing, Misra, LeCun, Deny | 2021 | SSL / Redundancy | [arXiv:2103.03230](https://arxiv.org/abs/2103.03230) |
| 24 | Masked Autoencoders Are Scalable Vision Learners **(MAE)** | He, Chen, Xie et al. | 2021 | SSL / Masked modelling | [arXiv:2111.06377](https://arxiv.org/abs/2111.06377) |

### Section 8 — Generative models

| # | Paper | Authors | Year | Category | Link |
|---|---|---|---|---|---|
| 25 | Auto-Encoding Variational Bayes **(VAE)** | Kingma & Welling | 2013 | Generative / Latent | [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) |
| 26 | Unsupervised Representation Learning with Deep Convolutional GANs **(DCGAN)** | Radford, Metz, Chintala | 2015 | Generative / GAN | [arXiv:1511.06434](https://arxiv.org/abs/1511.06434) |
| 27 | Neural Discrete Representation Learning **(VQ-VAE)** | van den Oord, Vinyals, Kavukcuoglu | 2017 | Generative / Discrete | [arXiv:1711.00937](https://arxiv.org/abs/1711.00937) |
| 28 | Progressive Growing of GANs for Improved Quality, Stability, and Variation | Karras, Aila, Laine, Lehtinen | 2017 | Generative / GAN | [arXiv:1710.10196](https://arxiv.org/abs/1710.10196) |
| 29 | A Style-Based Generator Architecture for GANs **(StyleGAN)** | Karras, Laine, Aila | 2018 | Generative / GAN | [arXiv:1812.04948](https://arxiv.org/abs/1812.04948) |
| 30 | Learning Transferable Visual Models From Natural Language Supervision **(CLIP)** | Radford, Kim, Hallacy et al. | 2021 | Multimodal / SSL | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) |
| 31 | Denoising Diffusion Probabilistic Models **(DDPM)** | Ho, Jain, Abbeel | 2020 | Generative / Diffusion | [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) |
| 32 | High-Resolution Image Synthesis with Latent Diffusion Models **(Stable Diffusion)** | Rombach, Blattmann et al. | 2021 | Generative / Diffusion | [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) |

### Section 9 — LLM scaling, alignment, and efficiency

| # | Paper | Authors | Year | Category | Link |
|---|---|---|---|---|---|
| 33 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Dao, Fu, Ermon, Rudra | 2022 | Efficiency | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |
| 34 | Training Compute-Optimal Large Language Models **(Chinchilla)** | Hoffmann et al. | 2022 | LLM / Scaling | [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) |
| 35 | Training Language Models to Follow Instructions with Human Feedback **(InstructGPT)** | Ouyang et al. | 2022 | Alignment / RLHF | [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) |
| 36 | LoRA: Low-Rank Adaptation of Large Language Models | Hu et al. | 2021 | Efficiency / Fine-tuning | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| 37 | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | Wei et al. | 2022 | Alignment / Prompting | [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) |
| 38 | Llama: Open and Efficient Foundation Language Models | Touvron et al. | 2023 | LLM / Open | [arXiv:2302.13971](https://arxiv.org/abs/2302.13971) |

---

### Why this reading order and not chronological?

- **AlexNet first** — readable cold; sets the motivation for everything
- **Dropout and Adam early** — training tools you need before dissecting any architecture
- **Word2Vec before Seq2Seq** — builds the representation intuition that carries into BERT and LLMs
- **VGGNet → BatchNorm → U-Net → ResNet** — each solves a problem the previous exposed
- **Seq2Seq → Bahdanau → Transformer** — tight causal chain from fixed bottleneck to attention-only
- **BERT → GPT-2 → GPT-3 → Scaling Laws** — language pretraining arc before vision transformers
- **ViT → Swin** — Swin is only legible after understanding ViT's limitations
- **SSL after ViT** — MoCo, DINO, and MAE all use ViT backbones
- **Generative section as a block** — VAE → GAN → VQ-VAE → DDPM → LDM is best read in one sitting
- **LLM scaling and alignment last** — all assume the transformer, pretraining, and GPT-3-scale context

---

*Plan maintained in Notion. Papers, step-by-step instructions, and StatQuest video mappings live in the linked pages.*
