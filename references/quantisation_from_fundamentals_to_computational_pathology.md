# Neural-network quantisation: from first principles to computational pathology

**A practical, literature-linked introduction for CS407**  
**Focus:** quantisation of image-classification models, then computational-pathology models and distilled students  
**Research snapshot:** 19 November 2026  
**Audience:** a computer-science student who knows basic Python/PyTorch and neural networks, but is new to quantisation and pathology

> **The central idea:** a neural network normally represents numbers using relatively high precision. Quantisation replaces some of those representations with a much smaller set of possible values. This can make a model smaller and, **if the deployment hardware actually supports efficient low-precision operations**, faster or less memory-hungry. The challenge is to preserve the useful information the model has learned.
>
> **The CS407 question:** Can we take a model that already works well, reduce its numerical precision—possibly after knowledge distillation—and obtain a useful improvement in *measured* inference cost without an unacceptable loss of predictive quality?

## How to read this guide

The first half is independent of medicine. Start at Sections 1–7 for the concepts, Sections 8–10 for method families, and Sections 11–12 for implementation and evaluation. Sections 13–17 introduce computational pathology from scratch, discuss prior work, and propose experiments for this repository. Section 18 is a glossary; Section 19 is an annotated reading list.

**Conventions:** **W8A8** means 8-bit weights and 8-bit activations; **W4A16** means approximately 4-bit weights with 16-bit activations. These are convenient descriptions of a *scheme*, not guarantees that every operation in the model uses those precisions. **FP32/FP16/BF16** are 32-bit/16-bit floating-point formats; **INT8/INT4** are 8-bit/4-bit integer formats. A **bit** is one binary digit, and eight bits make a **byte**.

---

# Part I — Why quantise?

## 1. Begin with an ordinary neural network

A neural network is a function that transforms an input, such as a picture, into an output, such as a predicted class. Its behaviour is controlled by **parameters**: learned numbers, most often called *weights* and *biases*.

Consider one simple layer:

$$
y = Wx + b.
$$

- **Input** `x`: a collection of numbers representing data from the preceding layer.
- **Weight matrix** `W`: learned numbers that determine how inputs are combined.
- **Bias** `b`: a learned offset.
- **Activation**: an intermediate number produced while the model runs, such as a component of `x` or `y`. The term is also used for nonlinear functions such as ReLU; here, *activations* means the intermediate tensors.
- **Tensor**: a multidimensional array of numbers. A scalar has zero dimensions, a vector has one, a matrix has two, and an image batch often has four: `[batch, channels, height, width]`.
- **Inference**: using an already-trained model to make predictions. **Training**: updating its parameters using data and an optimisation algorithm.

An image classifier might look like:

```text
Image (RGB pixel values)
      |
      v
Convolutional / transformer layers  [many weights and activations]
      |
      v
Class scores (e.g. cat, dog, bird)
      |
      v
Predicted class
```

A **class score**, often called a *logit*, is a raw output before conversion into probabilities by a function such as **softmax**. An image-classification model usually contains millions of learned parameters and must repeatedly multiply and add numbers to make one prediction.

### 1.1 Why do the numbers use so much memory?

A typical default is **FP32**, a 32-bit floating-point number. Floating point has a sign, exponent, and fraction (also called a significand), allowing a very large range of numbers and many approximate values within that range.

Imagine 25 million weights, without accounting for metadata:

| Representation | Bits per weight | Approximate raw weight storage |
|---|---:|---:|
| FP32 | 32 | 100 MB |
| FP16/BF16 | 16 | 50 MB |
| INT8 | 8 | 25 MB |
| INT4 | 4 | 12.5 MB |

Here, MB means **one million bytes**. These are *ideal raw-weight estimates*, not actual saved-model sizes. Real files can include bias tensors, quantisation scales, zero-points, padding, packing information, unquantised layers, or other metadata. INT4 must generally be **packed**—for example, two 4-bit values in one byte—to realise the intended storage reduction.

Notice an important difference: **quantisation does not usually remove parameters**. A model can still have 25 million weights after INT8 quantisation; it just stores them more compactly. By contrast, **pruning** removes or zeros selected weights/connections, and **knowledge distillation** trains a smaller *student* model to imitate a larger *teacher* model.

### 1.2 Why smaller might also be faster

Computer hardware performs calculations and moves numbers between memory levels. Often the time spent moving data is as important as the number of arithmetic operations.

Quantisation can help by:

1. **Reducing storage:** fewer bits per parameter in memory and on disk.
2. **Reducing data movement:** fewer bytes need to travel from main memory to a CPU/GPU compute unit. The **memory bandwidth** is the maximum rate of this transfer, usually expressed in GB/s.
3. **Using low-precision arithmetic:** CPUs, GPUs, and specialised accelerators may have fast INT8 or other reduced-precision instructions.
4. **Increasing possible batch size:** a **batch** is a group of inputs processed together. Lower memory usage may allow more inputs at once, improving **throughput**, the number of inputs processed per second.

However, none of these automatically implies lower latency. **Latency** is the elapsed time for one request or input. If an operation is unsupported, software may repeatedly convert between integer and floating-point formats, adding overhead. Small models may be limited by launch overhead or input decoding rather than calculations. An INT4 file stored on disk is not evidence that the model runs with efficient INT4 arithmetic.

**Mental model:** Quantisation is like packing objects into smaller boxes. It saves storage space, but delivery gets faster only if the trucks, roads, and unpacking process benefit too.

**Foundational reference:** [Jacob et al. (2018), *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html).

## 2. What exactly is quantisation?

**Quantisation** maps numbers from a large or continuous set of possible values to a smaller, discrete set. **Discrete** means the allowed values are individual steps rather than every point along a number line.

As an analogy, imagine recording temperature to the nearest 5°C instead of to 0.001°C. You need less information, but lose some precision:

```text
Original:  12.1°C    13.8°C    17.4°C    19.9°C
Rounded:   10°C      15°C      15°C      20°C
```

A quantised neural network does something similar to its weights, activations, or both.

Three terms are essential:

- **Quantisation error**: the difference between an original number and its quantised approximation.
- **Quantisation range**: the interval of values the chosen scheme can represent, such as an integer interval from -128 to 127.
- **Bit-width**: the number of bits used for each stored value. Fewer bits mean fewer representable values.

For an integer representation with `b` bits there are `2^b` possible bit patterns: 256 for 8 bits, 16 for 4 bits, and 4 for 2 bits. Signed and unsigned representations allocate those patterns differently. For a conventional signed INT8 format, the range is **-128 to 127**; for unsigned UINT8 it is **0 to 255**.

### 2.1 A worked example: symmetric uniform quantisation

**Uniform** means adjacent representable real values are separated by the same numerical distance. **Symmetric** means the real-valued range is approximately balanced around zero.

Suppose we pick a **scale** of `0.25`. The scale tells us how much one integer step represents in the original numerical units:

$$
q = \operatorname{clip}\!\left(\operatorname{round}(x/s), q_{\min}, q_{\max}\right),
\qquad \hat{x} = s q.
$$

Definitions:

- `x`: original real-valued number.
- `s`: positive scale (`0.25` here).
- `round`: round to a nearby integer using a stated rounding rule.
- `clip`: limit the value to a permitted interval.
- `q_min, q_max`: smallest and largest permitted integers.
- `q`: stored quantised integer.
- `x̂` (read “x-hat”): approximate original number after **dequantisation**, the conversion back to real-number units.

For these illustrative values, assume the integer range is wide enough to contain them:

| Original `x` | `x / 0.25` | Stored `q` | Reconstructed `x̂ = 0.25q` | Error `x̂ − x` |
|---:|---:|---:|---:|---:|
| -0.80 | -3.20 | -3 | -0.75 | +0.05 |
| -0.24 | -0.96 | -1 | -0.25 | -0.01 |
| 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 0.37 | 1.48 | 1 | 0.25 | -0.12 |
| 0.88 | 3.52 | 4 | 1.00 | +0.12 |

The approximation makes some values slightly different. What matters is whether these changes propagate through the network and change its predictions.

### 2.2 The two kinds of quantisation error

**Rounding error** occurs when a number falls between representable steps. With scale `0.25`, `0.37` becomes `0.25`, even though it is within the allowed range.

**Clipping error** (also called *saturation*) occurs when a value is outside the chosen range. If the greatest representable real value is `1.0`, a number such as `1.8` becomes `1.0`. This can be much worse than ordinary rounding.

Choosing the scale creates a trade-off:

- **Smaller scale**: fine numerical steps, but a narrower real-number range for a fixed bit-width; more values may be clipped.
- **Larger scale**: wider range, but coarser steps; more rounding error.

A **calibration** procedure is often used to choose this trade-off from representative data. We will define calibration in Section 6.

### 2.3 Affine or asymmetric quantisation

Symmetric quantisation is convenient when values include both positive and negative numbers. But some activations, such as those following a **ReLU** function (`ReLU(x) = max(0, x)`), are always nonnegative.

An **asymmetric** or **affine** quantiser adds a **zero-point** `z`:

$$
q = \operatorname{clip}\!\left(\operatorname{round}(x/s) + z, q_{\min},q_{\max}\right),
\qquad \hat{x} = s(q-z).
$$

The zero-point is the integer code used for real zero. It lets the quantiser allocate its limited codes to an interval that is not centred on zero. For example, if activation values lie between 0 and 6, using codes for a large negative range would waste representational capacity.

**Important:** Zero-point is not a trainable neural-network bias. It is part of the conversion between numerical representations.

### 2.4 Does the arithmetic really become integer arithmetic?

For a dot product—a sum of elementwise products—consider a simplified symmetric quantised case:

$$
x_i \approx s_x q_{x,i}, \qquad w_i \approx s_w q_{w,i}.
$$

Then

$$
\sum_i x_iw_i \approx s_xs_w \sum_i q_{x,i}q_{w,i}.
$$

The multiplication-and-summation in the middle can use integer operations. In practice, the individual products often accumulate in a wider format, such as **INT32**, so the sum does not overflow as easily. The final result may be **requantised** (converted into another low-bit representation) or returned to floating point. Real implementations must also handle zero-points, bias, rounding, and numerical saturation correctly.

**Kernel** means the implementation of a computation on particular hardware, such as a GPU matrix-multiplication routine. The efficient kernel, rather than the mathematical formula alone, determines actual speed.

---

# Part II — The major design choices

## 3. Which numbers are being quantised?

A neural network contains several kinds of numerical data. Not all are equally easy to compress.

| Scheme | What changes? | Intuitive benefit | Main complication |
|---|---|---|---|
| **Weight-only** | Model weights use low bit-width; activations remain at higher precision | Smaller checkpoints and weight-memory traffic | Arithmetic may still be high precision or require weight unpacking |
| **Weight-and-activation** | Weights and intermediate activations are low precision | Can benefit from efficient low-precision matrix operations and reduced activation traffic | Activations vary with inputs and can be hard to represent |
| **Selective or mixed-bit** | Some layers/tensors use fewer bits than others | Preserve sensitive components while compressing others | Requires choosing bit-widths and may have fragmented kernel support |
| **Embedding quantisation** | Model output features or cached features are compressed | Smaller stored feature datasets | Does not necessarily speed up the encoder itself |

A **feature** or **embedding** is a numerical representation created by a network, often a vector such as 1,024 numbers summarising an image. Embeddings become important in pathology, because thousands of tile embeddings may be saved per slide.

### 3.1 W8A8, W4A16, W4A8

The shorthand distinguishes weights (`W`) and activations (`A`). For instance:

- **W8A8**: 8-bit weights and activations, often a practical target for efficient integer computation.
- **W4A16**: 4-bit weights, 16-bit activations. Particularly useful when the bottleneck is loading weights from memory.
- **W4A8**: 4-bit weights, 8-bit activations. Potentially more aggressive, but harder to implement accurately and efficiently.

**Caveat:** A W8A8 model may still run layer normalisation, softmax, biases, residual additions, or its final output in a higher precision. Papers and implementations should state the layer-by-layer configuration.

### 3.2 Floating-point reduction versus integer quantisation

**FP16** and **BF16** are both 16-bit floating-point formats, not 16-bit integers. BF16 sacrifices some fine precision in exchange for a wide exponent range; FP16 has different precision/range trade-offs. **FP8** is an even narrower floating-point family, with multiple formats.

These formats are often discussed under **mixed-precision inference**, where different operations use different numerical formats. They may be excellent practical baselines, but FP16 is not the same intervention as INT8 quantisation.

For your dissertation, treat **FP32**, **FP16/BF16**, **INT8**, and **INT4** as distinct configurations. A claim such as “our model is quantised” must specify which tensors and which formats.

## 4. How many numbers share a quantisation scale?

A **scale** converts integer units to original real-number units. A quantiser may use one scale for many numbers or different scales for different groups.

Consider a weight matrix with 128 rows and 256 columns:

- **Per-tensor:** one scale for all 32,768 values. Simple and inexpensive, but a few unusually large weights may force a coarse scale for everything.
- **Per-channel:** one scale for each output channel (often a row of a linear layer or each output filter of a convolution). Different outputs can have different numerical ranges.
- **Per-group:** divide values into groups, such as blocks of 32 or 128 adjacent values, and store one scale per group. This is more flexible, particularly at 4 bits.
- **Per-token:** one scale for each sequence element's activation vector in a transformer, commonly used with dynamic activation quantisation.

A **channel** is a feature dimension produced by a layer (for example, one learned filter's output); a **token** is a sequence element processed by a transformer, such as a small image patch represented by a vector.

**Why finer scales help:** suppose one output channel has weights near `0.01`, while another has weights near `10`. Sharing a scale may erase the small values. Giving each channel its own scale preserves more detail.

**Why not always use a scale per number?** Each scale must be stored or computed, and many hardware kernels expect particular layouts. More scales can eat into compression gains or make execution inefficient.

## 5. Static, dynamic, and weight-only quantisation

These words describe *when* the quantisation parameters are decided, especially for activations.

### Static activation quantisation

Choose activation ranges and scales before deployment using representative example inputs. At inference, the scales are fixed.

**Motivation:** avoid recalculating scales for every new input and enable optimised deployment.

**Risk:** if the real inputs have a different numerical distribution from calibration data, some activations may clip or use the available range poorly.

### Dynamic activation quantisation

Determine activation quantisation parameters from the input encountered during inference (for example, for each token or batch). Weights may still be quantised ahead of time.

**Motivation:** handle variable activation ranges and reduce sensitivity to calibration choices.

**Cost:** computing scales during inference adds overhead. Whether the trade-off is favourable depends on the model and backend.

### Weight-only quantisation

Weights are fixed after training, so they can often be quantised *without an activation-calibration dataset*. Activations stay in a higher-precision format. This is a particularly approachable first experiment for transformer `Linear` layers, but it may provide limited benefit for models dominated by convolutions or other unsupported operations.

A `Linear` layer is a layer computing `y = Wx + b`. A **convolution** is a spatial filtering operation used by a convolutional neural network (**CNN**), such as ResNet.

## 6. Post-training quantisation (PTQ): the simplest main method

**Post-training quantisation (PTQ)** converts a model *after it has already been trained*. It does not require retraining the model weights, although some sophisticated PTQ methods optimise quantisation parameters or reconstruct layer outputs using calibration data.

```text
Existing trained FP32/BF16 model
               |
               v
   Choose low-precision scheme
     (e.g. W8A8, W4A16)
               |
               v
Collect representative activations
 (if the scheme requires calibration)
               |
               v
     Determine scales / ranges
               |
               v
Convert model to executable format
               |
               v
     Evaluate + benchmark
```

### 6.1 What is calibration?

**Calibration** is collecting statistics from example inputs to determine how values should be mapped to low-bit representations. You feed a model a small **calibration set**, observe the values at selected layers, and choose scales and clipping ranges.

The calibration set usually does **not** need labels. For example, the model can process 256 images even if you do not know which class each image belongs to. A **label** is a known answer used for supervised training or evaluation.

Different calibration strategies include:

- **Min–max:** capture the smallest and largest observed values and choose a range covering them. Simple, but susceptible to extreme outliers.
- **Percentile clipping:** ignore a small fraction of extreme values and allow them to clip, so common values get finer steps. Example: choose a high percentile rather than the absolute maximum.
- **Mean-squared-error (MSE) calibration:** choose a range to minimise average squared difference between original and reconstructed values. MSE gives larger errors more weight because they are squared.
- **Histogram or entropy-based calibration:** collect counts of values in intervals (**histogram bins**) and choose ranges that aim to preserve their statistical distribution; **KL divergence** is one possible measure of disagreement between two probability distributions.

No one calibration algorithm is best for every layer or dataset. The appropriate choice depends on the activation distribution, task, bit-width, and backend.

### 6.2 Why representative data matters

If calibration uses only bright, lightly stained images but deployment includes dark or differently stained images, the fixed scales may be inappropriate. This is especially relevant in pathology, where tissue-preparation and scanning conditions vary.

**Scientific rule:** calibration data belongs to the training/development side of the dataset split. Do not pick scales using the final test set, even though calibration is “unlabelled”. Looking at test inputs can still leak information from the held-out evaluation distribution.

### 6.3 PTQ strengths and weaknesses

**Strengths:** low additional training cost; easy to apply to existing checkpoints; makes controlled comparisons practical.

**Weaknesses:** quantisation error may become large with 4-bit or 2-bit values; some layers may be exceptionally sensitive; static schemes depend on representative calibration; backend limitations can prevent real speed-ups.

A good starting paper is [Nagel et al. (2021), *A White Paper on Neural Network Quantization*](https://arxiv.org/abs/2106.08295). The more advanced [BRECQ (Li et al., 2021)](https://arxiv.org/abs/2102.05426) improves PTQ by matching the outputs of blocks of the original and quantised network, rather than merely rounding every weight independently.

## 7. Quantisation-aware training (QAT): train while accounting for quantisation

**Quantisation-aware training (QAT)** simulates quantisation while the network trains or fine-tunes. **Fine-tuning** means continuing training from an existing checkpoint, usually with a smaller learning rate and/or a more focused dataset.

The intuition is that a model should learn to tolerate the numerical restrictions it will face at inference.

```text
Pretrained model
      |
      v
Insert simulated quantise/dequantise steps
      |
      v
Continue training with normal data/loss
      |
      v
Convert to a real low-precision deployment model
      |
      v
Evaluate + benchmark
```

### 7.1 What is fake quantisation?

A **fake-quantisation** operation rounds and clips a tensor as if it were low precision, then represents the rounded values with floating-point numbers so normal training software can operate on them. For example, it may change an FP32 value `0.37` to an FP32 representation of `0.25`.

This lets the model *experience quantisation error*. It does **not** mean the computation is actually happening with fast INT8 instructions.

### 7.2 How can a network learn if rounding is not differentiable?

Training usually uses **backpropagation**, which computes how a **loss function** changes when each weight changes. A loss function is a number measuring disagreement between a model's output and a desired result. Backpropagation relies on **gradients**, the derivatives of that loss with respect to weights.

Ordinary rounding has zero derivative almost everywhere, which makes it unsuitable for straightforward gradient-based learning. Many QAT implementations use a **straight-through estimator (STE)**: during the forward pass they apply rounding, but during the backward pass they approximate its derivative with a useful surrogate, often letting gradients pass through values in the quantisation range.

**Why QAT can recover accuracy:** the weights adapt to quantisation error rather than being rounded only at the very end.

**Why QAT costs more:** it needs training data, training time, optimiser settings, and careful handling of calibration and sensitive operations. It may not be worth it for 8-bit models whose PTQ performance is already almost unchanged.

### 7.3 Learned scales and learned clipping

Some QAT methods do not treat the quantiser itself as fixed:

- **LSQ — Learned Step Size Quantization:** learns the quantisation step size (`s`) using gradients, rather than setting it entirely through hand-designed statistics. [Esser et al. (2020)](https://arxiv.org/abs/1902.08153).
- **PACT — Parameterized Clipping Activation:** learns how strongly to clip activations before quantising them, balancing a smaller range against clipping error. [Choi et al. (2018)](https://arxiv.org/abs/1805.06085).

These are approaches *within* the broader QAT family, not separate goals from quantisation.

### 7.4 PTQ versus QAT, in one table

| Question | PTQ | QAT |
|---|---|---|
| Starting point | Trained model | Usually trained model plus additional fine-tuning |
| Learns model weights after quantisation is introduced? | Generally no | Yes |
| Needs labels? | Usually no for calibration | Usually yes for a supervised fine-tuning task; self-supervised QAT is also possible |
| Compute cost | Usually low | Higher |
| Particularly useful | A quick, reproducible INT8 baseline | Aggressive low-bit cases where PTQ loses too much accuracy |
| Common failure | Calibration/outliers/unsupported kernels | Training instability, inadequate data, or mismatch with deployment format |

---

# Part III — What makes some models difficult to quantise?

## 8. CNNs versus Vision Transformers

A **CNN**, such as **ResNet**, applies learned filters to nearby pixels and builds progressively more complex features. Its operations commonly include convolutions, **batch normalisation** (a learned rescaling/centering of features using training statistics), and nonlinear activations such as ReLU.

A **Vision Transformer (ViT)** divides an image into small patches, represents each as a token, and processes tokens using **attention**, which allows information from one token to influence another. Unlike a CNN, it does not rely exclusively on local sliding filters.

```text
CNN:  Image -> local convolutions -> deeper features -> class scores

ViT:  Image -> small patches -> token vectors -> attention blocks
                                                 -> class scores
```

**Attention** is an operation in which the model assigns context-dependent weights to different tokens. A transformer commonly computes three learned projections called **queries (Q)**, **keys (K)**, and **values (V)**, then calculates attention weights roughly as:

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^\top/\sqrt{d_k})V.
$$

Here, `d_k` is the key-vector dimension; the square-root term controls the magnitude of the scores. **Softmax** turns a list of scores into nonnegative numbers that sum to one. **Layer normalisation** standardises the features within a token and then applies learned scale/shift parameters.

### 8.1 Why ViTs can be harder to quantise

Some transformer components have awkward numerical distributions:

- **Outliers:** occasional values much larger than typical values. A shared quantisation range must either represent them or clip them.
- **Layer-normalisation outputs:** different channels may have very different distributions despite normalisation.
- **Softmax outputs:** values are restricted to `[0, 1]` but are often highly concentrated near zero, so a uniform grid may waste resolution.
- **Residual connections:** the model adds a layer's output back to its input. If the two branches use incompatible scales, additional conversions may be needed.
- **Layer sensitivity:** an early error may propagate through many later layers and significantly change final representations.

**Motivation for specialised ViT methods:** rather than assume every tensor can be uniformly quantised with one fixed scale, adapt the quantisation strategy to these difficult distributions.

### 8.2 Representative papers and what each changes

| Method | Basic insight | Why try it? | Read |
|---|---|---|---|
| **PTQ4ViT** | Uses a *twin uniform quantiser*—two uniform regions with different resolutions—to better approximate difficult transformer activations, alongside calibration choices | Simpler uniform quantisers may waste levels on attention-related distributions | [Yuan et al., ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7856_ECCV_2022_paper.php) |
| **RepQ-ViT** | Uses scale reparameterisation: first quantise difficult activations with schemes suited to their statistics, then algebraically redistribute scales so inference can use practical operations | Preserve accuracy without leaving impractical special quantisers in the deployment graph | [Li et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Li_RepQ-ViT_Scale_Reparameterization_for_Post-Training_Quantization_of_Vision_Transformers_ICCV_2023_paper.html) |
| **Q-ViT** | Quantisation-aware training designed for low-bit Vision Transformers, including quantisation-related training techniques | Lower bit-widths can require adaptation of network weights | [Li et al., 2022 preprint](https://arxiv.org/abs/2210.06707) |
| **BRECQ** | Reconstruct the output of network blocks after quantising them | Small per-weight errors do not always imply small model-output error | [Li et al., 2021](https://arxiv.org/abs/2102.05426) |
| **AdaRound** | Optimise which direction individual weights should round | Nearest-number rounding need not best preserve a layer's output | [Nagel et al., 2020](https://arxiv.org/abs/2004.10568) |

**Reparameterisation** means rewriting a computation into an algebraically equivalent form with differently distributed internal numerical values; ideally, the output before quantisation is unchanged. This can make the model easier to quantise or execute.

**Reconstruction** means comparing the outputs of the original and modified layers/blocks on representative inputs and adjusting quantisation choices to reduce disagreement. It is not reconstructing a medical image.

The papers in the table are primarily general-purpose computer-vision work. Their results should **not** be assumed to transfer unchanged to pathology encoders: pathology datasets, slide tasks, and hardware need their own evaluation.

## 9. Other useful method families and terminology

You will see these terms in the literature; they are useful to recognise even if they are not the first methods you implement.

### 9.1 Sensitivity analysis and mixed precision

**Sensitivity analysis** measures how much model quality deteriorates when a specific layer or block is quantised. A simple procedure is:

1. Start with the full-precision model.
2. Quantise only layer A, evaluate, and reset.
3. Quantise only layer B, evaluate, and reset.
4. Identify layers causing large losses.
5. Keep the sensitive ones at a higher precision.

A **mixed-precision** or **mixed-bit** model uses different precisions in different components. For example, most layers might use INT8 while a few sensitive layers remain FP16, or mostly INT4 with some INT8 layers.

**HAWQ** (Hessian-AWare Quantisation) is a more sophisticated example that uses estimates of a model's **Hessian**—second derivatives describing how rapidly the loss changes near its current weights—to inform precision allocation. The motivation is to spend limited bits where small numerical perturbations matter most. See [Dong et al., HAWQ](https://arxiv.org/abs/1905.03696).

### 9.2 Outlier-aware scaling

If one channel has very large activations, an INT8 grid shared with normal channels may be wasteful. **SmoothQuant** shifts quantisation difficulty between activations and weights by rescaling them in a way that preserves the mathematical operation in full precision. It was developed for large language models (LLMs), so pathology applicability is a research question rather than a demonstrated fact. See [Xiao et al., ICML 2023](https://proceedings.mlr.press/v202/xiao23c.html).

### 9.3 GPTQ and second-order weight-only PTQ

**GPTQ** is a post-training, usually weight-focused low-bit method that uses information about how rounding errors affect layer outputs to choose better approximations. It is especially associated with large transformer language models. Here **GPTQ** refers to a particular general-purpose quantisation method, **not** the unrelated 2026 paper name `GPTQ-MedFM`. See [Frantar et al., *GPTQ*, ICLR 2023](https://arxiv.org/abs/2210.17323).

### 9.4 Binary, ternary, and extremely low bit-widths

**Binary** representations use two possible values (approximately one bit); **ternary** representations use three (for example, `-1`, `0`, `1`). These can be highly compact but often require special training, architectures, and kernels. For this project they are better treated as possible later research rather than the first practical baseline.

### 9.5 Fake quantisation, simulated quantisation, and true deployment

These three conditions must not be confused:

1. **Numerical simulation:** rounded values approximate a low-bit model, possibly stored in FP32 tensors. Useful for studying *accuracy*.
2. **Packed low-bit weights:** smaller model storage, but the actual arithmetic path may unpack/dequantise to higher precision. Useful for *size*; speed must be measured.
3. **Backend-supported low-precision execution:** an executable model actually uses supported low-precision kernels. This is what supports a measured claim about *speed*.

One paper may compare fake-quantised accuracy while another measures real exported INT8 latency. They are answering different questions.

## 10. Distillation plus quantisation: why combine them?

**Knowledge distillation (KD)** trains a smaller *student* model to imitate a larger *teacher*. It may teach the student to match class scores (**response/logit distillation**), intermediate features (**feature distillation**), or other teacher-produced information. The teacher is used while training but is normally absent from the final deployed student.

For example:

```text
Large teacher model
        |   teaches during training
        v
Small student (e.g. ResNet18)
        |   PTQ or QAT
        v
Small low-bit student used for inference
```

Distillation and quantisation attack different sources of cost:

- **Distillation:** fewer parameters or a computationally cheaper student architecture.
- **Quantisation:** fewer bits per parameter or activation, sometimes cheaper arithmetic.

The advantages can *compound*, but not necessarily multiply perfectly. A tiny student may already fit entirely in cache; quantising it further may bring little speed benefit. A student with delicate representations may also be more sensitive to low-bit errors.

### 10.1 Four essential comparison groups

For a fixed student architecture, compare:

| Group | Training | Deployment |
|---|---|---|
| **A: baseline student** | Supervised training on labels | Full precision |
| **B: KD student** | Labels plus teacher guidance | Full precision |
| **C: quantised student** | Same training as A, then PTQ | Quantised |
| **D: quantised KD student** | Same training as B, then PTQ | Quantised |

Optional further experiments: QAT from A or B; different bit-widths; selective high-precision layers; multiple teachers. Keep the training data, student architecture, and budgets controlled so the differences are interpretable.

**Important:** compare *both* accuracy and measured inference cost. It is possible for D to be more accurate than C but slower than C due to a different exported graph, or for both to have the same latency because data loading dominates.

**Prior work:** [Mishra & Marr (2017), *Apprentice*](https://arxiv.org/abs/1711.05852) and [Polino et al. (2018), *Model compression via distillation and quantization*](https://arxiv.org/abs/1802.05668) explore connections between low-precision models and teacher supervision. They do not settle the pathology-specific question.

---

# Part IV — Practical implementation and credible evaluation

## 11. PyTorch: start with the smallest experiment that teaches you something

There are two good first exercises: (1) implement the mathematical idea on a tensor, then (2) apply a supported quantisation recipe to a **trained** model and measure what actually happens.

### 11.1 Tiny numerical experiment (educational, NOT efficient inference)

```python
import torch


def fake_symmetric_quantise(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Simulate symmetric quantisation, returning a floating-point tensor.

    Educational only: not a production quantiser and not an INT8 kernel.
    """
    if bits < 2 or bits > 8:
        raise ValueError("This example supports 2 to 8 bits")

    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax  # symmetric interval; one signed integer code unused
    max_abs = x.detach().abs().max()
    if max_abs == 0:
        return x.clone()
    scale = max_abs / qmax
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q * scale  # dequantised back into floating point


w = torch.tensor([-0.8, -0.24, 0.0, 0.37, 0.88])
for bits in (8, 4, 2):
    w_hat = fake_symmetric_quantise(w, bits)
    print(bits, "bits:", w_hat, "max error:", (w - w_hat).abs().max())
```

What this demonstrates: reducing bit-width usually increases numerical error. What it **does not** demonstrate: lower memory usage or faster inference, because the returned tensor is still floating point. The illustrative symmetric range intentionally uses `[-qmax, qmax]`, not all bit patterns of signed two's-complement integers.

**Follow-up exercise:** pass a model's original and fake-quantised weights through the *same input*, then compare its output vectors and classification accuracy. Keep the original model untouched so you can restore and compare it reliably.

### 11.2 Real quantisation: look at the current backend

In current PyTorch documentation, quantisation development is centred on **TorchAO**. The official TorchAO inference reference lists options including INT8 weight-only, INT8 dynamic activations plus weights, and INT4 weight-only. The exact supported operations, shapes, devices, and kernels change between versions. Verify them **on the machine where the benchmark will run**.

Illustrative code matching the documented 2026 TorchAO API:

```python
import copy
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig

# trained_model is an existing, trained nn.Module already on a supported device.
# Make a separate model to preserve the baseline for comparison.
quantised_model = copy.deepcopy(trained_model).eval()

# Convert eligible Linear layers according to the backend's current recipe.
# This does not imply that every operation in a ResNet or ViT is INT8.
quantize_(quantised_model, Int8DynamicActivationInt8WeightConfig())

# Now run representative inference, inspect which layers changed,
# check numerical quality, and benchmark on the actual target hardware.
```

**Do not treat the code as a universal ResNet quantiser.** TorchAO's common inference `quantize_` workflow is especially oriented towards eligible `Linear` operations. Convolutions, attention implementations, and export targets may require a different supported pathway, such as PyTorch export-based quantisation or a backend such as TensorRT. The model may fail conversion or run slower on unsupported hardware. See the [TorchAO inference guide](https://docs.pytorch.org/ao/stable/workflows/inference.html), [`quantize_` reference](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.quantize_.html), and [PyTorch quantisation migration notice](https://docs.pytorch.org/docs/main/quantization).

For a **ResNet classification prototype**, a backend-tested convolutional **static INT8 PTQ** route is arguably a better *first full-model* benchmark than assuming a linear-only recipe covers its main compute. For a **ViT pathology encoder**, linear weight-only or dynamic-activation quantisation is a useful controlled first experiment; inspect coverage and expect the rest of the transformer to use other precisions.

### 11.3 Three checks before publishing a speed-up

1. **Verify the representation:** are quantised weights truly packed into low-bit storage, and which layers stayed FP16/FP32?
2. **Verify the runtime:** what operations actually execute in low precision? Inspect exported graphs or profiler/operator information where possible.
3. **Measure on your hardware:** under identical inputs and controlled settings, is the quantised model genuinely faster, smaller, or more memory efficient?

### 11.4 Saving and loading quantised models

Quantised tensor layouts and module types may differ from ordinary FP32 models. Saving only an old model architecture plus a naïve state dictionary may not be sufficient. A reproducible artefact should record the source checkpoint, quantiser configuration and version, tensor formats, backend, architecture, and reloading procedure.

**Artefact** means a saved experimental output: checkpoint, exported model, configuration, metrics file, or benchmark log.

## 12. How to evaluate quantisation properly

An **ablation study** changes one design decision at a time to see whether it caused the observed effect. A **baseline** is an established reference configuration used for comparison. **Reproducibility** means someone can rerun a study using documented data splits, code versions, settings, and random seeds. A **seed** fixes the starting point for random-number generators, reducing accidental variation across runs.

### 12.1 Quality measures

For the first ImageNet-style classification prototype:

- **Top-1 accuracy:** proportion of images whose highest-scoring class is correct.
- **Top-5 accuracy:** proportion of images whose true class appears among the five highest-scoring classes; meaningful only for at least five classes.
- **Accuracy drop:** a difference in *percentage points*, e.g. 80.0% to 79.2% is a **0.8 percentage-point** drop, not a 0.8% relative decrease.
- **Per-class recall:** for one class, the fraction of truly positive samples identified as that class. This is important when a rare but clinically important class is affected disproportionately.

For later pathology tasks, additional metrics appear in Section 14.

### 12.2 Cost measures

| Metric | Plain-English meaning | A subtlety worth remembering |
|---|---|---|
| **Parameter count** | Number of weights/biases | Usually unchanged by quantisation |
| **Model storage size** | Bytes of deployable model/checkpoint | Report actual packed/exported size, not only an idealised bit-width estimate |
| **Latency** | Time for a prediction | Separate single-tile latency from whole-slide end-to-end latency |
| **Throughput** | Tiles/images processed per second | Depends on batch size and input pipeline |
| **Peak memory** | Largest memory footprint during a run | Include activation and runtime allocations, not just weights |
| **FLOPs** | Approximate floating-point operations counted by a model profiler | Not a reliable measure of reduced-precision speed; INT8 kernels may perform comparable mathematical operations at different cost |
| **Energy** | Electrical energy used, often joules | Requires power measurement or carefully qualified estimation |

A **GPU** is a graphics processing unit, frequently used for neural-network computation. A **CPU** is a general-purpose processor. An **FPGA** is a field-programmable gate array: reconfigurable digital hardware that can implement specialised low-precision computation. The same model can behave very differently on all three.

**Power** is the rate of energy use, in watts (joules per second). **Energy** is power integrated over time, in joules or watt-hours. A reduction in latency does not guarantee an identical percentage reduction in energy, because power draw may change.

### 12.3 A fair benchmarking protocol

For every model variant:

1. Use the **same frozen validation/test split**, input resize, normalisation, and evaluation unit.
2. Benchmark the **deployable model only**, not the training graph containing a teacher.
3. Record device model, CPU/GPU, software versions, precision, **batch size**, image shape, memory limits, quantisation backend, and exact layer coverage.
4. **Warm up** the model: run it a few times before timing to avoid measuring one-off startup/compilation costs as steady-state latency.
5. Time repeated runs. Report **median** (middle observation) and **p95** (95th-percentile latency: 95% of measured runs are no slower than this value).
6. Synchronise GPU execution before stopping a timer, or use appropriate GPU event timing: GPU work is commonly **asynchronous**, meaning the CPU can move on before the GPU has finished.
7. Report both *model-only inference* and, separately, *end-to-end processing* including loading, decoding, pre-processing, and saving outputs.
8. Record all failed conversions and unsupported layers. Do not quietly discard the configurations that could not be deployed.

For quick experiments, start with **FP32 → FP16/BF16 → INT8 → INT4**, but choose actual schemes compatible with the architecture and backend. Evaluate only on a held-out development set while tuning. Open the final test set once the choices are fixed.

### 12.4 An illustrative results table (blank: not measured)

| Configuration | W/A bits | Top-1 | Disk MB | Batch-1 latency | Tiles/s | Peak GPU MB | Backend |
|---|---|---:|---:|---:|---:|---:|---|
| Original ResNet18 | FP32/FP32 | — | — | — | — | — | PyTorch |
| ResNet18 lower-precision | FP16/FP16 where supported | — | — | — | — | — | PyTorch |
| ResNet18 PTQ | INT8/INT8 where supported | — | — | — | — | — | Specify |
| ResNet18 QAT | INT8/INT8 where supported | — | — | — | — | — | Specify |
| ViT weight-only PTQ | INT4/FP16 where supported | — | — | — | — | — | Specify |

A dash means **no claim; measurement not yet made**. Do not fill it with values from unrelated papers or different hardware.

---

# Part V — Computational pathology, starting from zero

## 13. What is computational pathology?

**Pathology** studies disease, often by examining tissue. **Histopathology** specifically studies thin slices of tissue under a microscope to identify changes associated with disease. **Computational pathology** uses software, statistics, and machine learning to analyse digitised pathology images.

A common laboratory preparation is **H&E staining**, short for **haematoxylin and eosin**. Haematoxylin makes many cell nuclei appear blue/purple; eosin colours other tissue components pink. A **cell nucleus** is a structure inside a cell containing most of its genetic material. H&E colour patterns help pathologists distinguish tissue structures, cells, and possible abnormalities.

### 13.1 Whole-slide images (WSIs), tiles, and ViT patches

A **whole-slide image (WSI)** is a very large digital scan of a microscope slide. It can contain billions of pixels (**gigapixel scale**). Such an image is usually too large to send through a standard image classifier all at once.

So a computer divides it into manageable images called **tiles** or **crops**. Typical tile sizes might be 224×224 or 256×256 pixels, although real workflows vary.

A ViT then *again* divides an individual tile into smaller **ViT patches**, e.g. 16×16 pixels. The terms *tile* and *patch* are sometimes used interchangeably in pathology papers, which causes confusion. In this guide:

- **Slide** = the full digitised microscope specimen.
- **Tile** = one image cropped from the slide and given to the model.
- **ViT patch/token** = a smaller subdivision *inside that tile* used by a transformer.

```text
Tissue sample on a glass slide
              |
              v
Whole-slide image (gigapixels)
              |
          Tile extraction
              |
    +---------+---------+
    |         |         |
 Tile 1    Tile 2     Tile 3  ... thousands more
    |         |         |
    v         v         v
Tile encoder (CNN or ViT)
    |         |         |
    v         v         v
Embedding Embedding Embedding
    \         |         /
     \        |        /
       Slide aggregator
              |
              v
       Slide-level prediction
```

A **tile encoder** maps each tile into an embedding vector capturing visual features. A **slide aggregator** combines many tile embeddings to make a decision about the entire WSI. An **embedding dimension** is the number of values in a vector, such as 768 or 1,024.

### 13.2 Resolution, magnification, and MPP

Pathology images may be examined at different **magnifications**, such as 10×, 20×, or 40×, meaning the tissue is viewed at different apparent scales.

**MPP**, or **micrometres per pixel**, describes actual physical resolution: how many micrometres of tissue each pixel spans. A micrometre is one thousandth of a millimetre. For example, an MPP of 0.5 means each pixel represents about half a micrometre of tissue.

For fair evaluation, two models must see comparable *physical tissue context*, not merely images with equal pixel dimensions. A 224×224 image at one MPP may cover a different area of tissue than a 224×224 image at another MPP.

### 13.3 Why many pathology models use foundation-model encoders

A **foundation model** is a model trained on broad data to learn representations reusable across many downstream tasks. **Pretraining** is the initial learning stage on a large dataset; **downstream tasks** are later applications built upon those learned features.

A pathology foundation-model encoder can learn useful visual descriptions of tissue from many tiles, often using **self-supervised learning (SSL)**. SSL creates a training signal from the data itself instead of relying on a pathologist to label every image. **DINOv2** is a prominent visual SSL method; pathology models such as UNI and Virchow build on related self-supervised methods.

Why this helps: getting millions of expert tissue annotations is expensive, whereas many scanned tissue images can be used for representation learning without per-tile diagnostic labels.

Representative foundation models include **UNI** ([Chen et al., 2024](https://www.nature.com/articles/s41591-024-02857-3)), **Virchow** ([Vorontsov et al., 2024](https://www.nature.com/articles/s41591-024-03141-0)), and **Prov-GigaPath** ([Xu et al., 2024](https://www.nature.com/articles/s41586-024-07441-w)). They are *not interchangeable*: they differ in architecture, tile resolution, pretraining corpus, weights availability, and their output representations.

**Relevant to quantisation:** a large encoder may run once for **every tile**, making its compute and model memory costs important. But a slide aggregator can also be substantial, so it should not be ignored.

## 14. What do pathology models actually predict?

### 14.1 Tile-level classification

Predict one label for an individual cropped tile, such as tumour versus non-tumour. **Tumour** means abnormal tissue growth; whether it is benign or malignant depends on the task.

This resembles ordinary ImageNet classification, but the inputs are tissue images and the labels have medical meaning.

### 14.2 Whole-slide classification

Predict one label for the entire slide, such as the presence of a cancer subtype. Slides often have only a slide-level label, without knowing exactly which of the thousands of tiles contain the relevant feature.

**Multiple-instance learning (MIL)** is a common solution: learn from a **bag** (collection) of tiles with a single slide label. An **attention-based MIL aggregator** can give different tiles different importance when forming a slide-level decision. Here *attention* is a mechanism for learning weights over tiles; it is related in spirit but not necessarily identical to the full transformer self-attention formula.

**Why this matters to quantisation:** if a rare but decisive tile receives a slightly worse embedding after quantisation, the slide-level classifier may miss a diagnosis even when average embedding errors appear small.

### 14.3 Semantic segmentation

Predict a class for each pixel: for example, tumour, connective tissue, or background. **Semantic** means pixels of the same class share a label even if they belong to separate structures.

Common metrics:

- **IoU (Intersection over Union):** for one class, the number of pixels in both predicted and true regions divided by the number in either region.
- **mIoU:** mean IoU over the evaluated classes.
- **Dice coefficient:** `2 × overlap / (predicted size + true size)`, measuring region overlap; 1 is perfect, 0 is no overlap.

### 14.4 Instance segmentation, particularly nuclei

Separate individual objects, such as different cell nuclei, even if two nuclei share the same class. A **semantic** mask can tell you where nuclei are; an **instance** mask also distinguishes nucleus A from nucleus B.

A quantised model might achieve similar average pixel overlap but merge adjacent nuclei, so object-level errors can matter more than a single aggregate segmentation score.

### 14.5 Domain shift: different hospitals, scanners, and stains

**Domain shift** means the data a deployed model sees differs from the data used to develop it. Pathology images can vary due to staining, scanning equipment, specimen preparation, and institutions. This variability is not necessarily biologically meaningful.

Quantisation adds a possible extra sensitivity: **calibration statistics learned from one hospital's stain distribution may not be representative of another's**. This is a *hypothesis to evaluate*, not a universally proven quantisation effect.

Useful background: [*Towards robust foundation models for digital pathology* (2026)](https://www.nature.com/articles/s41467-026-73923-2) examines robustness to non-biological variation. It supports the need for external-domain evaluation, not a claim that a particular quantiser causes the same failures.

---

# Part VI — What has the literature already done in pathology?

## 15. The most important distinction: quantising model weights versus quantising representations

The word **quantisation** has several meanings in this field:

1. **Numerical model quantisation:** convert neural-network *weights and/or activations* into lower precision, e.g. FP32 → INT8. This is the primary focus of the CS407 quantisation workstream.
2. **Feature/embedding quantisation:** compress the embeddings that an encoder produces. The encoder can remain completely unquantised.
3. **Vector quantisation (VQ):** map an entire vector to a representative vector in a learned **codebook** (a collection of prototype vectors), storing a discrete code/index rather than each component independently. It can be used to compress or organise representations and is *not* the same as changing every model weight to INT8.

**Toy vector-quantisation example:** suppose a codebook contains four prototype tissue-feature vectors called `C0`, `C1`, `C2`, and `C3`. For each embedding, choose the closest prototype and store its index, e.g. `2`. Later, the index can be used to retrieve the prototype. Instead of preserving all original feature values exactly, the system preserves which characteristic prototype the feature most resembles. More sophisticated systems use multiple tokens, levels, or codebooks.

This difference is crucial when reading papers: a headline containing “quantisation” does not prove that a pathology foundation-model encoder executes using INT8/INT4 kernels.

## 16. An evidence-based guide to relevant studies

The following studies provide **direct pathology evidence**, **related medical-imaging evidence**, or **methodological foundations**. They are not all directly comparable.

### 16.1 Direct histopathology: joint sparsity and quantisation-aware training (2023)

**Paper:** Dina Aboutahoun et al., [*Lightweight Histological Tumor Classification Using a Joint Sparsity-Quantization Aware Training Framework*](https://doi.org/10.1109/ACCESS.2023.3327221), *IEEE Access*, 2023.

**Question:** Can histological tumour classification be made practical on lower-resource systems?

**Method in ordinary language:** combine **sparsity** (having many zero or absent weights/connections) with quantisation-aware training. Both reduce resource requirements, but they do so in different ways.

**Reported result:** the authors' abstract reports **94.26% classification accuracy** and approximately **5× reduction in average memory footprint** for their studied model and task. These numbers are *not* results for UNI, Virchow, or a general pathology foundation model, and are not guaranteed for your dataset.

**Why it matters:** quantisation of histology classifiers is not a completely untouched idea. The novel question is more specific: architectures, foundation-model representations, low-bit schemes, real deployment, or combinations with distillation.

### 16.2 Direct tissue/medical optimisation: post-training parameter quantisation (2025)

**Paper:** [*Optimization of deep learning models for inference in low resource environments*](https://doi.org/10.1016/j.compbiomed.2025.110615), *Computers in Biology and Medicine*, 2025.

**Question:** Can existing medical-imaging models be deployed more efficiently without extensive retraining?

**Method:** compare **graph optimisation** (rewriting or combining operations in an execution graph to run more efficiently) with **post-training parameter quantisation** (PT-PQ). They evaluate several medical tasks including **colorectal cancer segmentation on H&E tissue images**, as well as MRI and diabetic-foot imaging.

**Reported overall finding:** optimisations improved model runtime and memory properties while maintaining useful task performance in the studied settings; the article describes overall model-size reductions in the range of approximately **50–80%** across its optimised configurations. Those percentages should not be interpreted as a per-model guarantee or as a pathology-foundation-model benchmark.

**Why it matters:** establishes a clear precedent for post-training parameter quantisation in digitised tissue analysis, not just small toy classifiers.

### 16.3 Direct pathology foundation-model work: PathVQ (NeurIPS 2025)

**Paper:** Honglin Li et al., [*PathVQ: Reforming Computational Pathology Foundation Model for Whole Slide Image Analysis via Vector Quantization*](https://proceedings.neurips.cc/paper_files/paper/2025/hash/59f278de1619bdb6b53fd04e8e0976e0-Abstract-Conference.html), *NeurIPS*, 2025.

**Problem:** most pathology pipelines summarise each tile with a single representation, sometimes a **[CLS] token** (a learned transformer summary token). But information about the *spatial arrangement within a tile* may remain in the other tokens. Keeping all token features for every tile of a huge slide is costly.

**Method:** use **vector quantisation** and **multi-scale VQ**—representing information at more than one spatial scale—to compress patch-token representations into discrete indices, then decode/aggregate features for whole-slide learning. Here **decoder** means a component that reconstructs an approximate representation from compressed codes, not a text generator.

**Reported result:** the abstract describes a **64× reduction** in representation dimensionality (`1024 → 16`) for its feature-compression mechanism. This is *not* evidence that encoder weights have become 64× smaller or that its forward pass is 64× faster.

**Why it matters:** shows pathology-specific interest in compact representations and highlights a second potential optimisation target: *stored tile/patch embeddings* as well as *the encoder that produces them*.

### 16.4 Adjacent medical imaging: real INT8 deployment with MedPTQ (2026)

**Paper:** [*MedPTQ: a practical pipeline for real post-training quantization in 3D medical image segmentation*](https://pmc.ncbi.nlm.nih.gov/articles/PMC12912285/), *Journal of Medical Imaging*, 2026.

**Question:** Can simulated INT8 quantisation be converted into a real, faster model on GPUs?

**Method:** take trained FP32 PyTorch segmentation models, perform calibration, and deploy actual INT8 **TensorRT** engines. **TensorRT** is NVIDIA's inference compiler/runtime for optimising models on compatible GPUs.

**Reported result:** across studied 3D segmentation models, the paper reports approximately **3.2–3.8× smaller model sizes**, **2.0–2.7× faster latency**, and **1.1–1.5× lower GPU memory use**, with an absolute mean Dice change no greater than 0.006. These figures are **for 3D medical segmentation**, not specifically histopathology, and depend on the authors' hardware and models.

**Why it matters:** an excellent example of measuring *actual deployed* quantisation rather than accuracy under simulated rounding alone.

### 16.5 Adjacent medical foundation-model PTQ: Permutation-COMQ (2026)

**Paper:** Yineng Chen et al., [*Weight Group-wise Post-Training Quantization for Medical Foundation Model*](https://arxiv.org/abs/2604.07674), CVPR Workshops 2026.

**Problem:** low-bit weight quantisation is sensitive to outliers and differences across channels.

**Method:** **Permutation-COMQ** uses a weight-aware reordering strategy together with a post-training quantisation procedure designed to avoid costly backpropagation. **Permutation** means changing the order in which items are arranged; the goal is to make weight quantisation more manageable while preserving the needed computation.

**Relevance:** a recent example of low-bit PTQ research for medical foundation models. Treat transfer to pathology-specific ViTs as something to test, not assume. The paper's abstract reports favourable low-bit results in its experiments, but it does not establish a general speed-up on your models.

### 16.6 Adjacent medical foundation-model PTQ: GPTQ-MedFM (2026)

**Paper:** [*Generalized post-training quantization for medical image segmentation foundation model*](https://doi.org/10.1016/j.media.2026.104287), *Medical Image Analysis*, 2026.

**Problem:** some medical segmentation foundation models are very large, and calibration data may be scarce because medical data are difficult to share.

**Method:** a specialised PTQ approach, **GPTQ-MedFM**, seeks to reduce quantisation-induced error across different medical image segmentation foundation models. Its reported evaluation covers **six models** and **nine imaging modalities**, including different calibration scenarios.

**Relevance:** medical foundation-model PTQ and the robustness of calibration strategies are active research topics. Do not equate this paper with ordinary GPTQ for language models or assume it experimentally tested a particular pathology encoder unless its model list establishes that.

### 16.7 Adjacent uncertainty-aware work: quantising Bayesian models (2023)

**Paper/technical discussion:** [*Quantization for Bayesian Deep Learning: Low-Precision Characterization and Robustness*](https://zenodo.org/records/8271449) and the accompanying [Intel overview](https://www.intel.com/content/www/us/en/developer/articles/technical/quantization-framework-bayesian-deep-learning.html).

**What is a Bayesian neural network (BNN)?** A model that represents uncertainty in its weights or predictions using probability distributions rather than only one fixed value per weight. **Uncertainty** describes how sure a prediction is, which can matter in medical applications.

**Relevance:** the examples include histology-image classification and low-precision inference. It suggests further questions about whether quantisation changes model confidence or uncertainty estimates, although that is beyond the basic CS407 deliverable.

### 16.8 Pathology foundation-model distillation (background for combinations)

Pathology-specific *distillation*, though not necessarily integer quantisation, is already an active line of research:

- [Filiot et al. (2025), *Distilling foundation models for robust and efficient models in digital pathology*](https://arxiv.org/abs/2501.16239) studies smaller pathology models learned from larger ones.
- [Usuyama et al. (2026), *GigaPath-Flash and GigaTIME-Flash*](https://arxiv.org/abs/2607.18218) introduces efficient distilled models for pathology slide/tumour-microenvironment analysis.
- [Kaspar et al. (2026), *DistillPath*](https://arxiv.org/abs/2608.17872) studies a compact distilled pathology encoder.
- [Lv et al. (2026), *Computationally Efficient Pathology Segmentation using Knowledge Distillation from Foundation Models*](https://arxiv.org/abs/2609.03947) explores task-specific segmentation students.

**Takeaway:** a reasonable dissertation experiment is to measure how much **additional** benefit low-bit integer quantisation provides after distillation. Because 2026 literature is moving quickly, do not make an unqualified claim that “nobody has quantised pathology models” or that a combination is globally unprecedented.

## 17. What research questions are still meaningful?

A literature gap is not simply “no-one has ever used quantisation”. More defensible questions include:

1. **Architecture sensitivity:** do CNN students and ViT students react differently to identical INT8/INT4 precision targets on tissue tasks?
2. **Student versus teacher:** is a distilled student more or less robust to quantisation than the same architecture trained without a teacher?
3. **Calibration data:** how many tissue tiles are needed, and do calibration outcomes change across tissue types, stain conditions, hospitals, or scanners?
4. **Task sensitivity:** does slide classification remain stable when tile embeddings change? Does tissue or nuclear segmentation lose small clinically important boundaries?
5. **Where should we spend bits?** Can selected sensitive ViT layers or the first/last layers remain at higher precision with a better accuracy–cost trade-off?
6. **Deployment reality:** do quantised encoders actually accelerate whole-slide processing, or are I/O and tile generation the limiting factors?
7. **Compound compression:** do distillation and quantisation yield complementary gains, and how does this compare with simply choosing a smaller native architecture?
8. **Embedding storage:** what happens when the encoder itself is quantised, the saved embeddings are quantised, or both?

The first six are already useful even if the techniques themselves are established. The scientific value comes from a carefully controlled experiment in the *specific pathology setting*, not merely creating an INT8 checkpoint.

---

# Part VII — How this fits `tile-and-error/imagenet-classification`

## 18. Read the repository architecture before adding quantisation

Sources inspected:

- [Organisation](https://github.com/tile-and-error)
- [`imagenet-classification` PLAN.md, version 1.1, 17 Nov 2026](https://github.com/tile-and-error/imagenet-classification/blob/main/PLAN.md)
- [PR #4: scaffolded basic building blocks](https://github.com/tile-and-error/imagenet-classification/pull/4/files)
- [Scaffold branch: `lwfm/core/graph.py`](https://github.com/tile-and-error/imagenet-classification/blob/scaffold/lwfm/core/graph.py)
- [Scaffold branch: `lwfm/core/losses.py`](https://github.com/tile-and-error/imagenet-classification/blob/scaffold/lwfm/core/losses.py)
- The supplied **CS407 Project Specification**, dated 30 October 2026, particularly §§1.1.2, 1.2–1.3, 2, and 3.

**Plan versus current implementation:** the PLAN describes intended interfaces and future stages. PR #4 is an open scaffold of core building blocks, not evidence that Phase 3 quantisation is implemented. The scaffolded `ModelNode` wraps an ordinary `nn.Module`; `ExecutionGraph` connects nodes by named inputs/outputs and exposes `deployable_model(node_name)` to retrieve the underlying module. Those are useful integration points, but exported preprocessing and standalone inference must still be verified. The implementation can evolve after the date of this guide.

### 18.1 Key pipeline concepts, in plain English

- **`nn.Module`:** the standard PyTorch class defining a model/layer. Quantisation should work with ordinary modules rather than force every researcher to write a custom model type.
- **`ModelNode`:** a lightweight wrapper around a model that declares which named data it reads and which output names it produces.
- **`ExecutionGraph`:** a **directed acyclic graph (DAG)** of model nodes. “Directed” means information flows from earlier to later nodes; “acyclic” means there are no circular dependencies. It can run a student and, later, one or more teachers.
- **`LossBinding` / `LossComposer`:** connect output names to loss functions and combine losses, useful for KD training. These are not needed for a simple post-training conversion.
- **`Recipe`:** a planned ordered sequence of whole experimental operations, such as train, evaluate, quantise, evaluate again, and benchmark.
- **`Stage`:** one such operation. **`TransformStage`** is the planned wrapper for modifying a model.
- **`ModelTransform`:** the planned operation that receives a model and returns a transformed model and a report. PTQ naturally fits here.
- **`BenchmarkRunner`:** the planned way to measure cost on the final model rather than on the teacher+student training graph.
- **`ArtifactRef`:** a planned reference to a saved input or output artefact, so the experiment retains the exact model it measured.

### 18.2 Where should PTQ go?

PTQ acts on **an already-trained deployable model**, not on the training loss:

```text
[Load trained baseline checkpoint]
                |
                v
       [Evaluate baseline]
                |
                v
       [Benchmark baseline]
                |
                v
       [PTQ model transform]
       + optional calibration
                |
                v
    [Save converted model + config]
                |
                v
      [Evaluate quantised model]
                |
                v
      [Benchmark quantised model]
```

The `ModelTransform` can record:

```yaml
method: ptq
scheme: w8a8  # or w4a16 etc.
quantiser: torchao_or_other_named_backend
weight_precision: int8
activation_precision: int8
calibration_manifest: path/to/development_calibration.csv
excluded_layers: []
hardware: recorded_by_benchmark
source_checkpoint: path/to/immutable/model.pt
```

This is an **illustrative configuration**, not a ready-to-run YAML schema accepted by PR #4. A **manifest** is a file listing exactly which samples were used, usually with stable identifiers and paths.

### 18.3 Where should QAT go?

QAT changes training behaviour, so it belongs in a supported training/fine-tuning route plus a conversion/export step:

```text
[Load trained student]
         |
[Insert fake-quantisation / QAT setup]
         |
[Fine-tune using existing Trainer]
         |
[Convert to real inference format]
         |
[Evaluate and benchmark]
```

The PLAN anticipates a training **callback** (code invoked at an appropriate point during training) for schedules such as enabling certain quantisers or progressively reducing precision. It is better to start with a standard supported QAT API than to invent a complex generic callback framework before it has a real use.

### 18.4 Protect baseline models and experimental provenance

Because `ExecutionGraph.deployable_model("student")` exposes a module that **shares its weights with the graph**, an in-place quantiser could unexpectedly mutate the baseline. Quantise a **separate copy** or reload from an immutable checkpoint. Always preserve links from:

`original training artefact → quantisation config and calibration → quantised deployment artefact → evaluation and benchmark results`.

If a teacher is present, quantise **the deployable student**, not the teacher+student graph. Also check that the deployable module includes—or explicitly documents—the correct input pre-processing; otherwise you might benchmark an encoder that receives a different input from the trained system.

### 18.5 A staged research plan that fits the repo's phases

**Milestone 0 — Learn the mechanism (no repository changes needed).**

- Run the toy symmetric quantiser in §11.1 on a tensor and inspect error at 8, 4, and 2 bits.
- Quantise different layers independently on a small, pretrained classification network; plot layer-output differences. At this stage the experiment may be fake-quantised and should be labelled accordingly.

**Milestone 1 — Wait for a verified end-to-end baseline (Phase 1 gate).**

- Train/load ResNet18 on a small ImageNet-style subset and verify predictions, evaluation, checkpoint round-trip, and benchmarks.
- Freeze a clean baseline checkpoint and data manifests. The PLAN uses synthetic `FakeData` for code smoke testing and a real image subset for scientific checks. **Never use FakeData to claim accuracy.**

**Milestone 2 — First *real* PTQ result (Phase 3's first quantisation goal).**

- Select a real supported CPU or GPU backend for the primary ResNet operators.
- Apply INT8 PTQ, using training/development images for calibration if required.
- Confirm that the exported model actually uses the intended format.
- Run the same quality/latency/throughput/size tests as the baseline.
- Keep this as the first reproducible `ModelTransform` implementation; no QAT needed yet.

**Milestone 3 — Controlled quantisation variants.**

- Compare per-tensor versus per-channel scales where supported.
- Compare activation calibration methods and calibration sample counts.
- Compare dynamic versus static INT8 if there are valid backend routes for the model.
- Test INT4 weight-only for supported layers; note that this is not an identical full-network alternative to INT8 W8A8.
- Add layer-sensitivity analysis to identify where high precision needs to be retained.

**Milestone 4 — Combine with KD (Phase 2 + Phase 3).**

- Compare A/B/C/D groups from §10.1 with a common student architecture and a fixed test protocol.
- Test whether PTQ acts differently on a supervised student versus a distilled one.
- Add QAT only if PTQ visibly harms accuracy or if low-bit exploration is worth the additional compute.

**Milestone 5 — Pathology transfer.**

- Begin with a small tissue-tile classification task, then a WSI MIL task.
- Quantise the **tile encoder first**, because it runs on every tile. Keep the aggregator unchanged as a controlled baseline.
- Then study encoder + aggregator quantisation and, separately, quantised saved embeddings.
- Evaluate across tissue types/scanners/sites and, if available, segmentation tasks with pixel/object metrics.

### 18.6 A first experiment matrix worth actually running

| Study | Model | Intervention | Main question |
|---|---|---|---|
| E0 | ResNet18 | FP32 baseline | What does the trained system achieve? |
| E1 | ResNet18 | FP16/BF16 inference | How much can simple floating-point reduction achieve? |
| E2 | ResNet18 | Real INT8 PTQ | Are model size, latency and quality improved on the selected backend? |
| E3 | ResNet18 | Different calibration sets | How sensitive is INT8 to calibration sample selection? |
| E4 | ResNet18 | INT8 QAT if E2 loses accuracy | Is extra training justified? |
| E5 | Supported ViT | INT8 dynamic or weight-only | Are transformer layers more sensitive, and how much is actually quantised? |
| E6 | Same student | KD and KD+PTQ | Do improvements compound? |
| E7 | Pathology tile encoder | PTQ, then optionally QAT | Does performance transfer from ImageNet to histology? |
| E8 | Pathology WSI pipeline | Quantise encoder only versus full pipeline | Which component dominates slide-level cost? |

The order respects the repository's Phase 1/2/3 goals. You can **research** and prototype independently before Phase 3 is merged, but the codebase should not claim that unsupported stages already work.

### 18.7 Pathology-specific experimental traps to avoid

- **Patient leakage:** multiple slides from the same person should not be split between training and test. A model could learn patient-specific artefacts instead of general medical patterns.
- **Pretraining overlap:** if the pathology foundation model saw images from a benchmark dataset during pretraining, the result may be optimistic. External non-overlapping test sets are useful.
- **Stain/site bias:** vary calibration and test domains deliberately; do not assume performance on one scanner proves robustness.
- **Tile versus slide unit:** a tiny change in per-tile accuracy does not translate directly into the same change in slide-level prediction accuracy.
- **Inflated speed-up:** preprocessing, tile extraction, slide reading, attention aggregation, and I/O may dominate whole-slide latency.
- **Incompatible views:** if comparing a student and teacher, ensure they see the same underlying image content, even if their required normalisation/resolution differs.
- **Improper precision claims:** record which particular operations remained floating point.
- **Clinical significance:** report rare-positive recall and false negatives, not just an overall average. Research benchmarks alone do not establish clinical safety.

---

# Part VIII — Glossary and reading path

## 19. Compact glossary

| Term | Short definition |
|---|---|
| **Activation** | Intermediate tensor produced when the network runs |
| **Affine quantisation** | Quantisation using a scale and zero-point to represent ranges not centred at zero |
| **Attention** | A mechanism that combines information from multiple tokens using learned/context-dependent weights |
| **Backend** | Hardware-specific software pathway that executes model operations |
| **Batch** | Several samples processed together |
| **Bit-width** | Number of bits used to represent a value |
| **Calibration** | Using representative data to choose quantisation scales/ranges |
| **Channel** | A feature dimension or output filter of a neural-network layer |
| **Clipping** | Mapping a number outside the allowed range to the nearest boundary |
| **CNN** | Convolutional neural network, with learned filters operating over spatial neighbourhoods |
| **Codebook** | A set of representative vectors used in vector quantisation |
| **DAG** | Directed acyclic graph: connected operations with no circular dependencies |
| **Dequantisation** | Mapping a stored quantised value back to an approximate real value |
| **Distillation / KD** | Training a student network using information from a teacher network |
| **Embedding** | Vector of learned features describing an input |
| **Fake quantisation** | Rounding/clipping simulation without necessarily executing integer kernels |
| **FLOPs** | Estimated number of floating-point operations; not the same as measured inference time |
| **Foundation model** | Broadly pretrained model reusable across downstream tasks |
| **FPGA** | Reconfigurable hardware that can implement specialised computations |
| **FP32 / FP16 / BF16** | Floating-point numerical formats with different ranges and precision |
| **Gradient / backpropagation** | Quantity/procedure that tells training how to update weights to reduce loss |
| **H&E** | Haematoxylin and eosin, common purple/pink tissue stain |
| **Inference** | Making predictions with a trained model |
| **INT8 / INT4** | Integer numerical representations with 8 or 4 bits |
| **IoU / Dice** | Overlap measures commonly used for segmentation |
| **Kernel** | Hardware-specific implementation of an operation, such as matrix multiplication |
| **Latency** | Elapsed time to complete a prediction or request |
| **MIL** | Multiple-instance learning: making predictions from sets of tiles using bag-level labels |
| **Model transform** | Operation that modifies a model, e.g. PTQ |
| **MPP** | Micrometres per pixel, a measure of tissue-image physical resolution |
| **Outlier** | An unusually large or otherwise extreme numerical value |
| **Parameter / weight** | Learned number controlling a model's behaviour |
| **Per-channel / per-group** | Quantisation schemes where different channels/groups use different scales |
| **PTQ** | Post-training quantisation of an already trained model |
| **QAT** | Quantisation-aware training, where quantisation effects are represented during learning |
| **Requantisation** | Conversion from one quantised scale/format to another |
| **Residual connection** | Adding a block's input to its output, helping information/gradients flow |
| **Scale** | Real-number value represented by one quantised integer step |
| **Segmentation** | Predicting a region/class for pixels or distinguishing individual objects |
| **Stain variation** | Differences in how tissue colours appear across preparation and scanning conditions |
| **STE** | Straight-through estimator, a gradient approximation used for non-differentiable rounding |
| **Throughput** | Inputs (or tiles) processed per second |
| **Tile** | Image crop extracted from a much larger whole-slide image |
| **Token** | A transformer sequence element, often representing one ViT patch |
| **ViT** | Vision Transformer, which represents image patches as tokens and uses attention |
| **VQ** | Vector quantisation: map vectors to prototypes in a codebook |
| **W8A8 / W4A16** | Shorthand for weight and activation bit-widths |
| **WSI** | Whole-slide image: a digitised microscopy slide |
| **Zero-point** | Integer code that corresponds to real zero in affine quantisation |

## 20. Recommended reading, from easiest to most specialised

This is an annotated *selection*, not a claim that the literature review is exhaustive. Some items are peer-reviewed conference/journal papers, while others are preprints or technical documentation.

### First-principles quantisation

1. **Nagel et al. (2021), [*A White Paper on Neural Network Quantization*](https://arxiv.org/abs/2106.08295).** The best broad conceptual reference for quantisers, scales, calibration, PTQ, and QAT. Start here after this guide.
2. **Jacob et al. (2018), [*Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html).** Classic explanation of how integer arithmetic can actually implement a neural-network inference path.
3. **Esser et al. (2020), [*Learned Step Size Quantization*](https://arxiv.org/abs/1902.08153).** Good QAT reference focused on learning the scale.
4. **Nagel et al. (2020), [*Up or Down? Adaptive Rounding for Post-Training Quantization*](https://arxiv.org/abs/2004.10568).** Why simply rounding each weight to its nearest representable value is not always optimal.
5. **Li et al. (2021), [*BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction*](https://arxiv.org/abs/2102.05426).** How local output-reconstruction objectives improve lower-bit PTQ.

### Vision Transformers and low-bit methods

6. **Yuan et al. (2022), [*PTQ4ViT*](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7856_ECCV_2022_paper.php).** Addresses difficult ViT activation distributions using specialised quantisation.
7. **Li et al. (2023), [*RepQ-ViT*](https://openaccess.thecvf.com/content/ICCV2023/html/Li_RepQ-ViT_Scale_Reparameterization_for_Post-Training_Quantization_of_Vision_Transformers_ICCV_2023_paper.html).** A useful discussion of layer normalisation, softmax, and scale reparameterisation.
8. **Li et al. (2022), [*Q-ViT*](https://arxiv.org/abs/2210.06707).** Quantisation-aware techniques for low-bit ViTs; read when ordinary PTQ has limitations.
9. **Frantar et al. (2023), [*GPTQ*](https://arxiv.org/abs/2210.17323); Xiao et al. (2023), [*SmoothQuant*](https://proceedings.mlr.press/v202/xiao23c.html).** Transformer-oriented techniques from the language-model setting; conceptual inspiration, not automatic pathology solutions.

### Computational pathology groundwork

10. **Campanella et al. (2019), [*Clinical-grade computational pathology using weakly supervised deep learning on whole slide images*](https://www.nature.com/articles/s41591-019-0508-1).** Explains how slide classification can work without tile-level labels.
11. **Chen et al. (2024), [*Towards a general-purpose foundation model for computational pathology*](https://www.nature.com/articles/s41591-024-02857-3).** Introduces UNI and the importance of general-purpose tile representations.
12. **Xu et al. (2024), [*A whole-slide foundation model for digital pathology from real-world data*](https://www.nature.com/articles/s41586-024-07441-w).** Prov-GigaPath and tile-to-slide modelling.
13. **Pocock et al. (2022), [*TIAToolbox as an end-to-end library for advanced tissue image analytics*](https://www.nature.com/articles/s43856-022-00186-5).** Practical pathology-image processing and model tooling.

### Evidence relating quantisation to histology and medicine

14. **Aboutahoun et al. (2023), [*Lightweight Histological Tumor Classification Using a Joint Sparsity-Quantization Aware Training Framework*](https://doi.org/10.1109/ACCESS.2023.3327221).** Direct early example of quantisation in histology classification.
15. **[Optimization of deep learning models for inference in low resource environments](https://doi.org/10.1016/j.compbiomed.2025.110615) (2025).** Direct tissue-segmentation evidence alongside other medical workloads; uses post-training parameter quantisation and graph optimisation.
16. **Li et al. (2025), [*PathVQ*](https://proceedings.neurips.cc/paper_files/paper/2025/hash/59f278de1619bdb6b53fd04e8e0976e0-Abstract-Conference.html).** Distinct, pathology-specific **vector quantisation of features**, not model-weight INT8 deployment.
17. **[MedPTQ](https://pmc.ncbi.nlm.nih.gov/articles/PMC12912285/) (2026).** Clear demonstration of real exported INT8 acceleration for *3D medical segmentation*, with measured hardware results.
18. **Chen et al. (2026), [*Weight Group-wise Post-Training Quantization for Medical Foundation Model*](https://arxiv.org/abs/2604.07674).** Recent medical-model low-bit PTQ research.
19. **[Generalized post-training quantization for medical image segmentation foundation model](https://doi.org/10.1016/j.media.2026.104287) (2026).** Recent foundation-model PTQ across medical modalities and calibration conditions.
20. **[Towards robust foundation models for digital pathology](https://www.nature.com/articles/s41467-026-73923-2) (2026).** Robustness and non-biological variation; a rationale for cross-site evaluation.

### Practical APIs and reproducibility

21. **[TorchAO quantised inference documentation](https://docs.pytorch.org/ao/stable/workflows/inference.html)** and **[`quantize_` API](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.quantize_.html)**. Treat these as version-dependent references for actually implemented low-bit kernels.
22. **[Your repository PLAN](https://github.com/tile-and-error/imagenet-classification/blob/main/PLAN.md)**, especially *Phase 3 — Recipe transforms and combined compression*, plus its benchmarking and scientific-rules sections.
23. **Your CS407 Project Specification (30 October 2026)** (supplied separately as a PDF; no public URL assumed). Use its quantisation, evaluation, dataset, and contribution sections as the authoritative project scope.

---

## 21. Final mental model

If you remember only one sequence, make it this:

```text
1. A model has many floating-point weights and intermediate activations.
2. Quantisation represents some of those numbers using fewer levels/bits.
3. Fewer levels introduce rounding and sometimes clipping errors.
4. PTQ chooses a low-bit representation after training; calibration may help.
5. QAT teaches the model to tolerate rounding during fine-tuning.
6. CNNs and ViTs behave differently; some layers may need more precision.
7. A smaller checkpoint does NOT automatically mean faster inference.
8. In pathology, the model often runs on thousands of tiles from one slide.
9. Quantise the encoder, and measure what happens to downstream slide tasks.
10. Your research contribution is the measured accuracy–cost–robustness trade-off,
    especially when quantisation is combined with distillation.
```

**The best initial milestone:** one trained ResNet18, one clean full-precision baseline, one *actually executable* INT8 PTQ variant, and a trustworthy table showing quality, actual model size, latency, throughput, and hardware. That is enough infrastructure to begin asking interesting research questions instead of merely adding another compression option to a framework.

*Source and scope note (19 Nov 2026):* The repository PLAN and scaffolded PR, the supplied CS407 specification, official PyTorch/TorchAO documentation, and the linked research papers informed this guide. Numerical results attributed to papers belong to their own experimental settings. The proposed CS407 experiments and suggested architecture integrations are recommendations, not claims that those features or outcomes already exist. Research on the topic is active; re-check recent publications before making a formal novelty claim in a dissertation.
