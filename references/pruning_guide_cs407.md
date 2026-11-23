# Neural Network Pruning: From First Principles to Computational Pathology

**A beginner-friendly research guide for the CS407 lightweight foundation-model project**  
**Literature checked:** 19 November 2026  
**Companion sources:** [tile-and-error/imagenet-classification](https://github.com/tile-and-error/imagenet-classification), its [implementation plan](https://github.com/tile-and-error/imagenet-classification/blob/main/PLAN.md), [pull request #4](https://github.com/tile-and-error/imagenet-classification/pull/4), and the team's *Lightweight Foundation Models for Whole-Slide Image Representation Learning* project specification (30 October 2026).

> **How to use this guide.** Sections 1–6 explain pruning from scratch. Sections 7–9 introduce computational pathology and survey actual research. Sections 10–12 turn the reading into feasible experiments. A glossary and clickable bibliography are at the end. You do **not** need prior pathology knowledge. Some familiarity with Python and neural-network training is helpful, but key terminology is explained as it appears.

---

## Contents

1. [The central idea](#1-the-central-idea-can-we-remove-work-the-network-does-not-really-need)
2. [Neural-network background](#2-the-smallest-amount-of-neural-network-background-you-need)
3. [A taxonomy of pruning](#3-the-pruning-taxonomy-four-independent-decisions)
4. [Model-pruning methods](#4-model-pruning-methods-from-simple-to-research-level)
5. [Transformer-specific pruning](#5-transformer-specific-pruning-different-things-called-pruning)
6. [Evaluating pruning](#6-how-to-know-whether-pruning-actually-helped)
7. [Computational pathology from zero](#7-computational-pathology-from-zero)
8. [What pathology research has tried](#8-what-has-already-been-tried-in-pathology-a-literature-led-map)
9. [Reading order](#9-how-to-read-the-literature-in-a-sensible-order)
10. [Your CS407 research pipeline](#10-applying-pruning-to-the-tile-and-error--cs407-pipeline)
11. [Experimental pitfalls](#11-experimental-discipline-mistakes-that-can-undermine-the-results)
12. [Pruning workstream](#12-a-suggested-minimum-viable-pruning-workstream)
13. [Glossary](#13-glossary-of-recurring-terms)
14. [Reference library](#14-clickable-reference-library)

---

## 1. The central idea: can we remove work the network does not really need?

Imagine a neural network that has learned to recognise dogs, cats, and cars. It contains millions of **parameters**: numerical values, usually called **weights** and **biases**, that determine how it transforms an input image into a prediction.

Not every parameter, group of parameters, or intermediate calculation necessarily contributes equally to a useful prediction. Some may be redundant: removing them changes the output little, especially if we give the remaining network a chance to adapt.

**Pruning** is the process of identifying and removing parts of a trained network—or, in a broader sense, parts of the data it processes—to reduce computational cost while keeping useful performance.

Think of editing a long report. You can remove individual words, entire paragraphs, or whole sections. All three make the report shorter, but their consequences are different. A computer similarly benefits much more from removing an entire calculation than from merely replacing some numbers in that calculation with zero.

### Four things we would like to improve

- **Model size:** the memory or storage required to hold the model's parameters. A smaller checkpoint is easier to download and deploy.
- **Inference latency:** the elapsed time needed to make a prediction on an input. *Inference* means using an already-trained network, rather than teaching it.
- **Throughput:** the number of inputs processed per second. A model might process one image quickly but have poor throughput in large batches, or vice versa.
- **Memory use:** how much RAM or GPU memory is needed while the model runs. This includes its weights and its intermediate **activations** (values created inside the network during a forward pass).

These are **not interchangeable**. For example, a model with many zero weights can remain slow if standard hardware still performs the same dense matrix multiplications.

### A crucial distinction: two different meanings of pruning

1. **Model pruning:** remove or disable parts of the *network*, such as individual weights, convolutional filters, attention heads, or entire layers.
2. **Input/token/patch pruning:** stop processing some parts of an *input*, such as visually redundant tissue regions in a whole-slide image.

Both matter enormously to computational pathology. They solve different bottlenecks, and their results must not be confused.

**Useful first question for every pruning paper:** *What exactly was removed—and did the implementation really stop computing it?*

---

## 2. The smallest amount of neural-network background you need

### 2.1 How a network processes an image

A **neural network** is a composition of functions called **layers**. In a typical image classifier:

```text
Input image
    ↓
Early layers: detect simple visual patterns
    ↓
Deeper layers: combine patterns into useful representations
    ↓
Classification head: scores for possible classes
    ↓
Prediction
```

A **class** is one possible label, such as “cat.” A **classification head** is the final part of the model that produces a score per class. The scores before conversion into probabilities are called **logits**. A **softmax** function can turn logits into a set of class probabilities that sum to one.

During **training**, the network predicts labels on example images, measures mistakes using a **loss function**, and adjusts weights through **backpropagation** (a method of calculating how each weight contributed to the loss). A **gradient** is the derivative of the loss with respect to a weight: it indicates how a small weight change would affect the loss. An **optimizer**, such as SGD or Adam, uses gradients to update parameters.

A **validation set** helps us choose model settings and pruning decisions; a separate **test set** is reserved for assessing the final chosen approach. Repeatedly choosing a pruning method based on test performance gives an unfairly optimistic result.

### 2.2 Convolutional neural networks (CNNs), filters, and channels

A **convolutional neural network (CNN)** processes images with small learnable pattern detectors. Each detector is a **filter** (or convolutional kernel): a small array of weights applied across the image. The resulting grid of numbers is called a **feature map** or **output channel**.

For example, a layer might take **64 input channels** and produce **128 output channels**. Each output channel comes from a filter bank that examines all input channels. At fixed image size and kernel dimensions, its approximate multiplication cost is proportional to:

`input_channels × output_channels × kernel_height × kernel_width × output_height × output_width`.

This suggests two ways of pruning a convolution:

- Delete a few individual weight values: the filter *dimensions remain unchanged*.
- Delete an entire output filter: the layer produces fewer channels, so subsequent layers can often perform fewer operations too.

The second is normally more promising for ordinary GPU/CPU speedups, **provided that the connected layers are physically rewritten**.

**ResNet** (short for *Residual Network*) is a CNN family containing **residual connections**: paths that add an earlier activation to the output of later layers. These help train deep networks, but they create constraints for pruning. If two tensors (multidimensional numerical arrays) are added, their relevant dimensions must agree. Removing channels from one branch but not the other can break the network.

**Batch normalization** is a layer that rescales and shifts activations, typically with a learned scale parameter often written as `gamma`. Some structured-pruning methods use small learned scales as clues that a channel is unimportant.

### 2.3 Vision Transformers (ViTs), patches, tokens, and attention

A **Vision Transformer (ViT)** is an image-processing model based on the Transformer architecture. Rather than using only convolutional filters, it commonly divides an image into small rectangular **patches** (for example, 16×16 pixels). Each patch becomes a vector of numbers called a **token embedding**, or simply a **token** when the distinction is clear.

The ViT processes these tokens using repeated **Transformer blocks**. A block usually contains:

- **Self-attention:** each token can combine information from other tokens; the learned strengths of these interactions are attention weights.
- **Attention heads:** several attention calculations performed in parallel, potentially learning different relationships.
- **MLP (multilayer perceptron):** a small feed-forward neural network applied to tokens. An MLP is built from learnable linear transformations and nonlinear functions.

A **CLS (classification) token** is an optional special token used to collect information for classification. **Positional embeddings** encode where image patches came from, because a plain sequence of token vectors does not automatically carry two-dimensional position information.

Self-attention's computation often grows roughly **quadratically in the token count**: if the number of tokens doubles, the pairwise attention component can become roughly four times as expensive (other operations have different scaling). This makes token pruning particularly interesting.

---

## 3. The pruning taxonomy: four independent decisions

When a paper says “we prune the network,” ask four separate questions.

### 3.1 **What is removed?** (granularity)

| Granularity | What disappears? | Intuitive example | Likely practical consequence |
|---|---|---|---|
| **Unstructured** | Arbitrary individual weights | Zero out 30% of numbers inside filters | Sparse weights; may **not** speed up ordinary dense kernels |
| **Semi-structured** | Weights in a constrained pattern | Keep 2 of every 4 weights in a supported operation (**2:4 sparsity**) | Can help on compatible sparse hardware and kernels |
| **Structured: filters/channels** | Entire CNN filters and their resulting channels | Change a layer from 128 to 96 output channels | Smaller dense layers; promising speedup |
| **Structured: neurons/MLP dimensions** | Entire intermediate features in fully connected layers | Reduce a Transformer MLP's hidden width | Smaller matrix multiplications if downstream shapes change |
| **Structured: attention heads** | Complete attention branches | Use fewer of 12 heads | Potentially lower cost when projections are genuinely reduced |
| **Structured: blocks/layers** | Entire network sections | Remove 4 of 12 Transformer blocks | Shallower model; potentially substantial speedup |
| **Token/patch pruning** | Pieces of the input or intermediate token sequence | Keep 500 of 1,000 patch tokens | Less input processing / smaller attention problem; model weights may be unchanged |

**Sparsity** is the fraction of entries or units that have been removed or set to zero. At 60% weight sparsity, 60% of considered weights are zero. For structural pruning, specify the denominator carefully: 60% of channels removed is not necessarily 60% of total parameters removed.

**Keep rate** is the fraction retained. For simple token selection, a 70% keep rate means 30% of tokens are discarded.

### 3.2 **How is importance judged?** (selection criterion)

A pruning **criterion** or **heuristic** is a rule for deciding what matters. Examples include weight magnitude, average activation, influence on the loss, or the measured accuracy drop when a component is temporarily disabled.

No criterion is universally correct. A small weight can be useful when its inputs are large; two individually useful components can still be redundant because they do nearly the same job.

### 3.3 **When does pruning happen?** (schedule)

- **Post-training pruning:** start with a trained model and prune it afterward. This is an attractive first experiment.
- **Pruning during training:** progressively remove capacity while the network learns.
- **One-shot pruning:** remove the desired amount in a single step.
- **Iterative or gradual pruning:** remove some, retrain or fine-tune, then remove more. **Fine-tuning** means continuing training from existing weights, generally to restore accuracy or adapt to a task.
- **Static pruning:** one reduced model or mask is used for every input.
- **Dynamic or input-adaptive pruning:** different images can retain different tokens, channels, or blocks depending on their content.

### 3.4 **Where is the budget applied?** (scope)

- **Local/layer-wise pruning:** each layer has its own pruning quota, such as 20% of weights per layer.
- **Global pruning:** all eligible weights or groups compete for a single overall budget. Some layers can lose many entries and others very few.
- **Uniform pruning:** apply the same fraction across eligible layers.
- **Non-uniform pruning:** allow different fractions, possibly to protect sensitive layers.

These terms overlap but are not exact synonyms. For example, a method can be **structured + global + iterative + post-training**, all at once.

---

## 4. Model pruning methods, from simple to research-level

### 4.1 Random pruning: your sanity-check baseline

**Idea:** randomly choose some weights or filters to remove, without using any information about importance.

**Method:** choose a pruning rate, use a fixed random seed (so the experiment is reproducible), select components, remove or mask them, and evaluate or fine-tune.

**Why bother?** It tests whether a clever method is actually better than removing arbitrary parts. If magnitude pruning barely beats random pruning, your importance score may not provide much value.

**Limitation:** random pruning can remove genuinely important components and hurt accuracy, especially at high pruning rates.

### 4.2 Magnitude pruning: remove small weights

**Idea:** a weight with a value close to zero might affect the output less than a weight with a large value. This is a convenient approximation, not a guarantee.

Suppose we have six weights:

`[0.8, -0.02, 0.5, 0.003, -0.7, 0.01]`

At 50% unstructured magnitude pruning, remove the three weights with the smallest **absolute value** (ignore their signs):

`[0.8, 0, 0.5, 0, -0.7, 0]`

The model now has 50% zeros **in those six positions**, but it still contains a six-element array. Ordinary dense multiplication may execute exactly as before.

**Procedure:** train → compute `abs(weight)` → select the smallest weights globally or per layer → apply a **binary mask** (1 = retain, 0 = disable) → optionally fine-tune → evaluate.

A mask acts element by element: `effective_weight = original_weight × mask`. For a fair experiment, make sure pruned values remain zero during fine-tuning rather than silently regrowing.

**Motivation:** simple, inexpensive, historically effective, and excellent for understanding how much redundancy a model has.

**Limitation:** weight magnitude does not directly measure output or loss importance, and the sparse pattern often needs special hardware/software for speedups. See [Han et al. (2015)](https://arxiv.org/abs/1506.02626), [Zhu & Gupta (2017)](https://arxiv.org/abs/1710.01878), and the [PyTorch pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial).

### 4.3 Filter and channel pruning: remove whole computations

**Idea:** instead of zeroing scattered numbers, delete a complete CNN filter and the corresponding output feature map.

Imagine a convolution with 64 input channels and 128 output channels. You decide 32 output channels contribute relatively little:

```text
Before: Conv(64 → 128 channels)
After:  Conv(64 →  96 channels)
```

The next convolution used to receive 128 channels and must now be changed to receive 96. Its relevant weights must be removed too. This operation is often called **structural surgery**.

A simple score is the **L1 norm** of each filter: add up the absolute values of all weights in that filter. A different score is the **L2 norm**: take the square root of the sum of their squares. Prune the lowest-scoring filters, modify dependent layers, and fine-tune.

**Why is this better for speed?** The surviving model uses *smaller dense tensors*, which ordinary GPUs and CPUs already know how to process efficiently.

**Challenges:** filter scores need not correlate with predictive importance, and **residual connections**, **skip connections** (paths that bypass intermediate layers), and concatenations (joining tensors along a dimension) create dependencies. A filter is rarely safe to remove in isolation.

Read [Li et al., *Pruning Filters for Efficient ConvNets* (ICLR 2017)](https://arxiv.org/abs/1608.08710) and [Fang et al., *DepGraph: Towards Any Structural Pruning* (CVPR 2023)](https://github.com/VainF/Torch-Pruning).

### 4.4 Network slimming: teach channels to advertise their usefulness

Recall **batch normalization**, which has a learned scaling number `gamma` per channel. If a channel's gamma becomes close to zero, that channel has little influence after scaling, so it becomes a plausible pruning candidate.

**Network slimming** encourages small scale values during training using a **regularization penalty**. Regularization means adding a preference to the training objective beyond simply fitting the labels. Here the preference is for some channel scales to become small.

The process is:

1. Train while penalising the sum of absolute channel scales.
2. Rank channels by the magnitude of those learned scales.
3. Remove low-ranked channels and fix dependent layers.
4. Fine-tune the smaller model.

The approach deliberately creates a network in which channel importance is easier to identify. Its limitation is that it changes the training recipe, and learned scale does not guarantee clinical relevance. See [Liu et al., *Learning Efficient Convolutional Networks Through Network Slimming* (ICCV 2017)](https://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html).

### 4.5 Activation-based pruning: look at what the network actually does

A **weight** is a stored parameter; an **activation** is a value produced when the model processes a particular input.

A filter could have moderate weights but almost always output nearly zero. Rather than judging its parameters, activation-based pruning measures the filter's outputs on a **representative dataset** (images resembling the ones expected at deployment).

For each channel, calculate a statistic such as mean absolute activation across example inputs. Prune channels with low scores.

**Motivation:** a filter's practical usefulness depends on both its weights and the data flowing through it.

**Limitation:** the score depends strongly on the examples used. A filter that rarely activates may detect an uncommon but diagnostically important abnormality. A representative *calibration set* is a small set used to measure model behaviour; it need not be used to train new weights. It should include relevant tissue types, stains, scanner variation, and rare classes where possible.

### 4.6 Gradient/Taylor pruning: estimate how much the loss would change

A **Taylor approximation** estimates how a function changes after a small change to its inputs. For a parameter `w`, one simple first-order pruning score is proportional to:

`abs(w × dLoss/dw)`

Here `dLoss/dw` is the gradient of the loss with respect to that weight. A large score suggests setting the weight to zero could substantially affect the training objective.

For a filter, variants calculate a similar score over its output activations and gradients, then aggregate it over data.

**Why use it?** Magnitude asks “how large is the number?” Taylor-based pruning asks “how much might removing it affect what the network is trying to achieve?”

**Limitations:** gradients depend on the loss, data, and current model state; first-order approximations can be inaccurate for large removals. See [Molchanov et al., *Pruning Convolutional Neural Networks for Resource Efficient Inference* (ICLR 2017)](https://research.nvidia.com/publication/2017-04_pruning-convolutional-neural-networks-resource-efficient-inference).

### 4.7 Second-order pruning: account for interactions and curvature

The **second derivative** describes how quickly the gradient changes. The collection of second derivatives is a **Hessian matrix**. It gives information about the curvature of the loss landscape around a trained solution.

Early work argued that whether a weight matters depends not only on its magnitude or gradient but also on how other weights can compensate when it is removed:

- **Optimal Brain Damage (OBD):** uses an approximation based on second-order sensitivity to rank parameter removal.
- **Optimal Brain Surgeon (OBS):** uses richer second-order information and allows compensating adjustments to surviving weights.

**Motivation:** pruning is not just deleting a value; the remaining parameters may adapt to preserve the function.

**Trade-off:** computing or approximating the Hessian is expensive, particularly for very large models. It is useful background but not the best first CS407 implementation. See [LeCun et al. (1989)](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html) and [Hassibi & Stork (1992)](https://papers.neurips.cc/paper_files/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html).

### 4.8 Learned gates and L0-style sparsity: let the model choose what to remove

A **gate** is a learned switch that controls whether some weight, channel, or token contributes to the computation. In a simple version, it is either zero (off) or one (on).

We can train the network to balance accuracy against the number of active gates:

`total_objective = prediction_loss + lambda × complexity_penalty`

`lambda` controls how strongly we value lower complexity. The **L0 count** is the number of nonzero elements; mathematically it encourages truly absent components, not just small ones. Binary gates are difficult to optimise using ordinary gradients, so researchers use differentiable approximations such as **hard-concrete gates**.

**Motivation:** instead of picking a heuristic after training, make removing unnecessary capacity part of learning.

**Trade-off:** more complicated training and extra choices about the penalty, gates, and how the final physical model is exported. See [Louizos, Welling & Kingma (ICLR 2018)](https://arxiv.org/abs/1712.01312).

### 4.9 One-shot versus iterative pruning; prune–fine-tune–repeat

**One-shot example:** remove 50% of eligible filters once, then fine-tune. Fast to experiment with, but can damage a network abruptly.

**Iterative example:** remove 10% of the *currently remaining* filters, fine-tune, measure, and repeat. The network gets opportunities to adapt, but experimentation takes longer.

Important arithmetic: removing 10% of the remaining filters five times does **not** mean exactly 50% total removal. The retained fraction is `0.9^5 ≈ 59%`, so about 41% have been removed. To reach a desired final retained fraction `r` over `k` equal relative rounds, keep `r^(1/k)` per round.

**Progressive or gradual pruning** can also increase sparsity smoothly through a training schedule rather than through discrete prune–fine-tune cycles.

**Lottery Ticket Hypothesis:** Frankle and Carbin investigated whether a large randomly initialised network contains smaller subnetworks that, when trained with suitable initial weights, perform comparably to the original. **Weight rewinding** is related: after identifying a sparse subnetwork, reset retained weights to an earlier training checkpoint, then retrain. This is an important research idea but distinct from simply making a pretrained model faster at inference. See [Frankle & Carbin (ICLR 2019)](https://dspace.mit.edu/entities/publication/d2335291-d7a7-418b-8708-0f47209827d6) and [Gale, Elsen & Hooker (2019)](https://research.google/pubs/the-state-of-sparsity-in-deep-neural-networks/).

### 4.10 Semi-structured N:M pruning

**N:M sparsity** means keeping exactly N weights in each eligible group of M weights, such as **2:4**, where two of every four weights survive.

Why compromise between unstructured and structured? It preserves more freedom than removing whole channels but uses a regular pattern potentially supported by dedicated sparse matrix hardware. **Kernel** here means an optimised software routine that carries out an operation on a CPU/GPU; sparse speedup requires a kernel and hardware that exploit the exact pattern.

**Caution:** the fact that a GPU supports some sparse format does not prove that your chosen convolution, model, precision, or PyTorch execution path will run faster. Benchmark the actual deployed operator.

### 4.11 Pruning *before* training: find a smaller network at initialization

**Initialization** means the starting (usually random) values of a model's weights, before it has learned from examples. Most of the techniques above begin with a trained model. Another family tries to identify a useful sparse network **at initialization**, potentially saving training computation as well as inference cost.

- **SNIP (Single-shot Network Pruning based on Connection Sensitivity):** estimates how sensitive the loss is to switching individual connections off, using a small amount of data and gradient information at initialization. It then chooses a sparse set of connections before ordinary training begins. See [Lee, Ajanthan & Torr (ICLR 2019)](https://arxiv.org/abs/1810.02340).
- **GraSP (Gradient Signal Preservation):** aims to preserve the network's ability to propagate informative training gradients after pruning. The emphasis is not simply on the immediate change in loss, but on maintaining the ability to *learn*. See [Wang, Zhang & Grosse (ICLR 2020)](https://arxiv.org/abs/2002.07376).

**Motivation:** perhaps we do not need to train the full dense model before finding a smaller model. **Limitation:** early pruning decisions can be unreliable, and unstructured sparsity still may not accelerate standard dense hardware. These are advanced baselines, not immediate necessities for the project's initial ResNet prototype.

### 4.12 Movement pruning: pay attention to how weights change during fine-tuning

**Movement pruning** is a gradient-informed method developed for adapting large pretrained models. Rather than only asking how large a weight is *now*, it tracks whether training is moving that weight towards contributing more or less, and uses learned importance scores to decide which connections survive.

**Motivation:** when adapting a pretrained model to a new task, a weight that starts near zero might become important, while a once-large weight might cease to matter. **Caution:** this approach was developed and demonstrated mainly in language-model fine-tuning, so its benefits should not be assumed to transfer directly to pathology CNNs or ViTs. See [Sanh, Wolf & Rush, *Movement Pruning* (NeurIPS 2020)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/eae15aabaa768ae4a5993a8a4f4fa6e4-Abstract.html).

---

## 5. Transformer-specific pruning: different things called “pruning”

### 5.1 Attention-head pruning

A Transformer block may have several parallel **attention heads**, each learning relationships among tokens. If two heads learn redundant information, perhaps one can be removed.

**Procedure:** score each head (for example by measuring the validation-loss change when disabled or by comparing its behaviour with others), remove less useful heads, adjust the projection matrices (the learned transformations that implement attention), then fine-tune if needed.

**Critical detail:** *masking* a head's output is not the same as physically eliminating the corresponding projection computation. Some head-pruning approaches do not produce proportional speedups unless the model's dimensions and implementation are carefully rebuilt.

Foundational reference: [Michel, Levy & Neubig, *Are Sixteen Heads Really Better than One?* (NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/2c601ad9d2ff9bc8b282670cdd54f69f-Abstract.html).

### 5.2 Transformer block pruning

A **block** is a repeated processing unit, usually including attention and an MLP. Block pruning removes complete blocks, reducing network depth.

For example:

```text
Before: input → B1 → B2 → B3 → B4 → B5 → output
After:  input → B1 → B2 ─────→ B4 → B5 → output
```

The remaining blocks normally keep compatible input/output dimensions, which can make block pruning simpler than arbitrary head-width changes. But layers are not functionally interchangeable: removing a block may erase a representation useful for certain tasks. Results depend on which block is removed and what recovery training or task-specific head is used.

### 5.3 Token pruning: process fewer image patches inside a ViT

A ViT might receive 196 image-patch tokens (14×14 patches), plus a possible CLS token. A token-pruning module scores tokens and discards some before later Transformer blocks.

```text
196 patch tokens
      ↓  importance selector
128 retained patch tokens
      ↓  expensive later attention blocks
classification
```

**Static token selection:** predefine which positions or patches to keep.  
**Dynamic token selection:** score each input image and retain different tokens for different images.

**Common scoring methods:** attention-based importance, learned lightweight scoring networks, token similarity, uncertainty, or downstream task relevance.

**Why it can work:** some regions are less useful for a particular decision, while attention cost increases rapidly with token count.

**Why it can fail:** a tiny or unusual region may be decisive. If it is discarded early, later blocks cannot recover its information. Learned selectors also cost computation and can be inaccurate.

An influential method is [DynamicViT (Rao et al., NeurIPS 2021)](https://proceedings.neurips.cc/paper_files/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html), which learns token-importance predictions and progressively shortens the token sequence. [A-ViT (Yin et al., CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Yin_A-ViT_Adaptive_Tokens_for_Efficient_Vision_Transformer_CVPR_2022_paper.html) adjusts token computation to image complexity.

### 5.4 Token merging is **not** exactly token pruning

**Pruning** discards a token entirely. **Merging** combines similar tokens into a summary so some information survives.

Example:

```text
Four similar tissue tokens → one representative token
```

Merging may protect useful information compared with dropping tokens. However, mixing two visually similar regions could still erase subtle differences important in pathology.

This distinction will matter when reading whole-slide research such as DTC-WSI and WSIR² below.

### 5.5 A helpful comparison

| Strategy | Reduces stored weights? | Reduces computation? | Biggest caveat |
|---|---|---|---|
| Individual-weight masking | Usually no, with ordinary dense storage | Not automatically | Zeros still run through dense kernels |
| Physical channel/filter pruning | Yes | Often, if dependencies are fixed | Residual and skip connections |
| Physical Transformer block pruning | Yes | Often | Loss of useful representation depth |
| Attention-head masking only | Not necessarily | Not necessarily | Projections may still be computed |
| ViT token pruning | No, usually | Yes, in later token-dependent layers if tokens are removed | Important token accidentally discarded |
| WSI tile selection *before* expensive encoding | No, usually | Can reduce tile-encoder work dramatically | Selector's cost, missing rare diagnostic areas |
| Token merging | Usually no | Can reduce sequence processing | Can blur information between regions |

---

## 6. How to know whether pruning actually helped

### 6.1 Accuracy versus efficiency: a trade-off, not one score

Suppose an unpruned image classifier gets 80% top-1 accuracy at 10 milliseconds per image. One pruned model gets 79.8% at 6 milliseconds; another gets 76% at 3 milliseconds.

Neither is automatically “best.” The choice depends on the application. We should report both predictive quality and cost and, ideally, plot them against each other.

A **Pareto frontier** contains models for which there is no alternative that is **at least as accurate and at least as efficient**, with a strict improvement on one of those two dimensions. Models on the frontier represent useful trade-offs.

### 6.2 What to measure

- **Top-1/top-5 accuracy:** the true ImageNet label is the first-ranked prediction or among the five highest-ranked predictions.
- **Parameter count:** how many parameters remain in the actual deployed model. Do not confuse zeros with removed parameters.
- **Stored checkpoint size:** bytes of the exported, inference-only model, specifying precision and file format.
- **FLOPs:** an estimate of floating-point arithmetic operations per inference. A decrease suggests less arithmetic, but does not prove a wall-clock speedup.
- **Latency:** measured elapsed time for a specified input, batch size, device and precision. Include warm-up runs (initial runs discarded to avoid one-time setup costs) and synchronise GPU measurements correctly.
- **Throughput:** images or tiles per second at a stated batch size.
- **Peak GPU memory:** maximum memory used during the measured inference workflow.
- **End-to-end WSI cost:** time from obtaining a slide, reading its tiles and performing inference to obtaining a final slide prediction. This includes I/O (reading data from disk), preprocessing, selection, encoding and aggregation.

**Do not claim an energy or carbon improvement from FLOP reduction alone.** Actual energy consumption depends on hardware, runtime, utilisation, data movement, and more.

### 6.3 A useful example of misleading “compression”

A 10-million-parameter, 32-bit model has about 40 MB of raw weight data (ignoring file-format overhead). Set half its weights to zero:

- **Logical sparsity:** 50%.
- **Dense parameter storage:** still approximately 40 MB.
- **Number of dense matrix operations:** often unchanged.

Physically remove half the parameters *in a valid smaller architecture*, and dense weight storage becomes approximately 20 MB. But even then, latency is an empirical question: some operations are memory-bound, small kernels have overhead, and GPUs respond differently to altered tensor shapes.

---

## 7. Computational pathology from zero

### 7.1 What is pathology, and what does the computer see?

**Pathology** is the study of disease. **Histopathology** studies disease by examining tissue under a microscope. The word **computational pathology** means using computer methods, including machine learning, to analyse such images and associated data.

A tissue specimen (a sample taken from a person) is prepared on a glass slide and stained to make its structures visible. A common stain is **H&E (haematoxylin and eosin)**: it tends to make cell nuclei appear blue-purple and many surrounding structures appear pink. A **nucleus** is the DNA-containing part of a cell. **Stroma** is supportive tissue around cells and tumour structures.

A digital slide scanner produces a **whole-slide image (WSI)**: a very high-resolution digital image of the tissue on the slide. These images can be so large that feeding all the pixels through an ordinary neural network at once is impractical.

Therefore, researchers divide slides into small **tiles**—also called *patches* in much of the pathology literature—and process them separately. The term **magnification**, such as 20× or 40×, describes an imaging scale. **Microns per pixel (MPP)** more precisely expresses resolution: how much physical tissue length corresponds to one image pixel. Lower MPP means higher physical resolution.

```text
Glass slide → scanning → whole-slide image (WSI)
                                ↓
                         tiles/patches
                                ↓
                image encoder (CNN or ViT)
                                ↓
                 feature vectors/embeddings
                                ↓
                   slide-level aggregator
                                ↓
                 slide-level prediction
```

An **encoder** turns an image into a compact numerical **embedding** or **feature vector** describing what it has seen. A **slide-level aggregator** combines information from many tile embeddings into a prediction for the entire slide.

### 7.2 What do pathology models predict?

**Tile-level classification:** classify a small region, such as distinguishing a tissue pattern.  
**Slide-level classification:** assign a label to the entire slide, such as a cancer subtype.  
**Semantic segmentation:** label individual pixels by category, such as tumour versus stroma.  
**Instance segmentation:** identify individual objects—even touching ones—such as separate cell nuclei.  
**Nuclei classification:** label each identified nucleus by cell type.  
**Biomarker prediction:** predict a measurable biological characteristic, potentially linked to disease or treatment.  
**Prognosis:** predict a likely future disease outcome, not merely classify what is visible now.

**Multiple-instance learning (MIL)** is especially common for WSIs. In MIL, a slide is a **bag** of tile **instances**. We may know the overall slide diagnosis but not which individual tile contains the decisive evidence. An MIL model learns to combine tile features using operations such as averaging, maximum pooling, or **attention pooling** (assigning learned importance weights to tiles).

**Weak supervision** means training with less precise labels than the ideal output would require—for example, using slide labels without tile-by-tile tumour annotations. MIL is a common response to this problem.

### 7.3 What is a pathology foundation model?

A **foundation model** is a model trained on large, broad datasets and intended to be reused for many later tasks. In pathology, models such as **UNI** or **Virchow2** learn representations of tissue images. They can often act as pretrained tile encoders for new classification or segmentation problems.

**Pretraining** is the broad initial training step; **downstream tasks** are the more specific problems addressed later. **Self-supervised learning** learns from images without needing every example to have a human-supplied diagnostic label; training targets are derived from the data or from different views of it.

Some models operate mainly on individual tiles; others also contain a slide-level encoder that combines many tiles. This distinction determines where the computational bottleneck lies.

### 7.4 Why pruning could matter more here than in everyday image classification

Imagine a WSI with 20,000 tissue tiles. Even a small extra cost per tile is multiplied thousands of times. Three distinct places could be optimised:

**A. Before tile encoding:** select only some tiles. This can avoid invoking the expensive tile encoder at all for discarded tiles.

**B. Inside the tile encoder:** prune CNN channels or ViT blocks/tokens so every retained tile is cheaper to process.

**C. During slide aggregation:** prune/merge tile embeddings or attention tokens so the model combining all tiles works on a shorter sequence.

These can be **combined**, but savings are not automatically additive. The selector itself has a cost; a cheaper tile encoder changes the value of early tile selection; and cached embeddings mean reducing encoder cost may not improve the measured slide-aggregator runtime.

### 7.5 The pathology-specific danger

A slide may be mostly normal tissue with a tiny region of cancer. An average-looking region could still contain a rare finding. Removing such regions may improve average benchmark accuracy while harming sensitivity to rare but important disease.

**Sensitivity/recall** measures the fraction of true positive cases correctly identified. **Specificity** measures the fraction of true negative cases correctly identified. In some clinical uses, missing a positive case (a **false negative**) is much more serious than an extra false alarm (a **false positive**).

Other important problems include **domain shift** (the deployed data differ from the training data), stain variation, scanner differences, and **data leakage** (information from test patients or slides influencing training or method selection). Tissue tiles from the same patient must not casually appear in both train and test splits.

---

## 8. What has already been tried in pathology? A literature-led map

The papers below address genuinely different types of pruning. Read the “what was reduced?” column before comparing any numbers. Reported results are the authors' findings, not independently reproduced results in this guide.

### 8.1 Earlier work on pruning weights and CNN channels

**[2019: *Breast cancer histopathology image classification through assembling multiple compact CNNs*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6805574/)**

- **Task:** classification of breast tissue microscopy images, including experiments using the **BreaKHis** and **BACH** datasets (collections of labelled breast histology images).
- **What was reduced?** CNN channels in a compact image-classification network.
- **How?** A **Squeeze-Excitation-Pruning (SEP)** block learns per-channel attention-like scale factors. The authors average these factors across training images to estimate channel importance, prune lower-scoring channels, and retrain. **Squeeze-and-Excitation (SE)** is a CNN mechanism that uses information across a feature map to learn which channels to amplify or suppress.
- **Why is it interesting?** The authors explicitly use pathology data to decide which channels to keep. This is an early example of structurally compressing histology classifiers, rather than just changing numerical precision.
- **Caution:** the full method also includes multiple-model ensembling—combining predictions from several trained networks—which may improve accuracy but increase deployment cost. Separate *single-model pruning* effects from *ensemble* effects.

**[2022: *Deep Neural Network Pruning for Nuclei Instance Segmentation in H&E-Stained Histological Images*](https://arxiv.org/abs/2206.07422)**

- **Task:** detect and individually outline nuclei. Experiments use **MoNuSeg** (multiple organs) and **TNBC** (triple-negative breast cancer) histology images.
- **What was reduced?** Individual network weights, using **unstructured magnitude pruning**.
- **How?** Compare **layer-wise** and **network-wide/global** magnitude pruning; perform iterative pruning with retraining. The system combines a **semantic segmentation branch** (where nuclei are) with a **distance-map regression branch** (predicted distances used to separate touching nuclei). **Regression** means predicting continuous numbers rather than a discrete class.
- **Reported result:** the authors report that, for their settings, 80% of weights could be pruned with under about 2% degradation in final instance-segmentation quality; some intermediate tasks tolerated even higher sparsity.
- **Critical caveat:** this is evidence that many weights may be unnecessary for prediction, **not** proof of an 80% real-world latency reduction. Their speedup analysis depends on sparse hardware support. Their segmentation measures include **Dice** (overlap score), **AJI** (Aggregated Jaccard Index), and **PQ** (Panoptic Quality, assessing instance detection and segmentation quality).

**[2024: Adnan et al., *Structured Model Pruning for Efficient Inference in Computational Pathology*](https://arxiv.org/abs/2404.08831)**

- **Task:** nuclei instance segmentation/classification using **HoVer-Net** on **PanNuke**, and colorectal tissue tile classification with ResNet18. HoVer-Net uses horizontal/vertical distance information to help separate individual nuclei; PanNuke contains annotated nuclei from different tissue types.
- **What was reduced?** Whole convolutional filters/channels, physically reducing network structure.
- **How?** Compare **L1**, **L2**, and network-slimming importance scores; one-shot, non-uniform, and iterative pruning; account for dependencies across U-Net-like encoder/decoder and residual connections.
- **Reported result:** the paper reports substantial compression with small performance losses. For iterative HoVer-Net pruning it reports around **80% lower latency (approximately 5–6× faster)** in its experimental setup at very high filter sparsity, alongside a substantial reduction in parameter count.
- **Why this should be a priority read:** it is a direct blueprint for *actual model inference efficiency* in computational pathology, and it highlights why pruning a complex segmentation architecture is harder than pruning a plain stack of convolutions.
- **Caution:** it is a 2024 research preprint; reproduce measurement conditions before translating its speedup to another model or GPU.

### 8.2 Pruning attention heads and pathology foundation models

**[2025: *Vision Transformers for Histopathological Image Classification with Efficient Head Pruning*](https://doi.org/10.1016/j.procs.2025.09.348)**

- **Task:** breast histopathology image classification with a Vision Transformer.
- **What was reduced?** Transformer attention heads.
- **How?** The study combines **knowledge distillation (KD)**—a smaller **student** model learns from the outputs of a more capable **teacher** model—with head-pruning decisions informed by head **confidence** and **similarity**. Confidence is a measure of how strongly a predictor favours a decision; similarity refers here to different heads exhibiting redundant behaviour.
- **Reported finding:** some head removal maintained or improved accuracy and reduced training time in the reported experiments.
- **Caution:** this does not establish that physically pruned inference kernels outperform ordinary ViTs on all hardware. Distillation and pruning are both used, so the benefit of each needs to be isolated experimentally.

**[2026: Hasan, Yeafi & Islam, *TAP-Path: Task-Adaptive Structural and Token Pruning for Efficient and Trustworthy Pathology Foundation Models*](https://arxiv.org/abs/2609.04071)**

- **Task:** pathology classification using the pretrained **Virchow2** tile encoder, with internal and external evaluation.
- **What was reduced?** **Transformer blocks** and **image-patch tokens** *inside* the encoder. This is different from removing whole WSI tiles before the encoder is called.
- **How?** Evaluate which blocks are dispensable for a chosen classification task using development/validation data; physically remove selected blocks; score and retain informative tokens; recover information from features taken at several surviving depths. **Task-adaptive** means the compression choices are chosen with a particular end task in mind, not assumed to work identically for every application.
- **Reported result:** retain **24 of 32 blocks** and **70% of patch tokens**; encoder parameters fall from **631.24 million to 473.70 million (−24.96%)**, and reported **analytical encoder compute** falls by **35.20%**. The paper also reports competitive classification on its benchmarks.
- **Key limitation:** **analytical FLOPs are not measured wall-clock latency**. The paper is a recent **preprint**; verify reproducibility, whether actual inference kernels exploit token reduction, and generalisation before treating numbers as deployment guarantees.
- **Why it matters to CS407:** this is especially close to the idea of taking an existing large pathology encoder and directly pruning it without first designing a smaller student architecture.

### 8.3 Whole-slide tile/token selection: related, but a different bottleneck

**[2024: *Creating an atlas of normal tissue for pruning WSI patching through anomaly detection*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10873359/)**

- **What was reduced?** Candidate WSI tissue patches considered by a downstream analysis or retrieval workflow.
- **Idea:** learn what normal tissue tends to look like, and filter out uninformative normal-looking patches so subsequent analysis focuses on potentially abnormal regions. **Anomaly detection** means identifying examples that depart from a learned normal pattern.
- **Why interesting:** it attacks redundancy in the **data**, rather than shrinking the network.
- **Risk:** “normal-looking” is task-dependent; some benign tissue context matters, and removing rare abnormalities would be dangerous. Its main demonstrated use concerns pathology image retrieval, not every slide-level classification task.

**[2026: *A deep learning framework for efficient pathology image analysis* — EAGLE](https://pmc.ncbi.nlm.nih.gov/articles/PMC13324285/)**

- **What was reduced?** The number of WSI tiles subjected to expensive detailed feature extraction.
- **How?** **EAGLE (Efficient Approach for Guided Local Examination)** uses a pretrained **CHIEF** model for relatively inexpensive tissue representation/selection, then applies the powerful **Virchow2** encoder to selected informative tiles.
- **Why interesting:** it makes explicit that, if there are thousands of tiles, choosing *which* tiles to encode may save more work than shrinking the encoder by a few percent. The paper evaluates a wide range of downstream pathology tasks.
- **Caution:** this is **guided tile selection**, not proof that the Virchow2 parameter count has been reduced. The first-stage selector itself has a cost, which should be counted.

**[2026: Li et al., *Whole slide image redundancy reduction for efficient pathological diagnosis* — WSIR²](https://doi.org/10.1016/j.bspc.2025.109289)**

- **What was reduced?** Redundant tokens in the WSI slide-level classification pipeline.
- **How?** Use cross-attention to identify **top-k** informative patches—meaning the *k* highest-scoring candidates—and merge less informative tokens, instead of independently processing every token at full cost.
- **Why interesting:** joint **pruning plus merging** may preserve more useful evidence than simply dropping every low-ranked patch.
- **Caution:** the paper is about sequence reduction in a WSI pipeline, not necessarily physically eliminating weights from a pretrained tile encoder.

**[2026: Rahman et al., *DTC-WSI: Dynamic Token Compression for Whole-Slide Images* (MIDL 2026)](https://proceedings.mlr.press/v315/rahman26b.html)**

- **What was reduced?** Tile-feature tokens entering or passing through a slide-level **MIL** computation.
- **How?** A lightweight importance network and multi-stage compressor combine **bipartite similarity matching** (matching similar tokens across two groups for merging) with **soft differentiable pruning** (smooth training-time gates that permit gradient-based learning). At inference, the system uses deterministic token compression.
- **Reported result:** **5–10× fewer tokens**, **up to 5.3× faster inference**, and **20–40% lower memory use** under the authors' benchmarks, alongside reported classification gains.
- **Lesson:** a hard early delete may be inferior to gradual, information-preserving compression under weak slide-level supervision. Its token reduction is not equivalent to a 5–10× reduction in tile-encoder parameters.

**[2026: Bozkurt et al., *SLICEChat: Progressive In-Encoder Token Pruning for Whole-Slide Pathology Language Models*](https://arxiv.org/abs/2609.24894)**

- **Task:** answer questions about whole-slide pathology images using a **multimodal large language model (MLLM)**—a model combining images with text and language generation.
- **What was reduced?** Visual tokens inside a slide encoder before subsequent multimodal processing.
- **How?** Progressively prune coherent tissue regions using learned selectors. A **region-aware** approach chooses groups of spatially neighbouring patches rather than unrelated individual patches. Language supervision helps decide which image evidence may be useful to a textual question.
- **Why interesting:** it explicitly treats spatial relationships between tissue regions as important when reducing the sequence.
- **Caution:** this September 2026 work is a **preprint**; its project page advertises WACV 2027, which is **future** relative to this guide's date. Do not describe it as already published at that conference.

### 8.4 What the literature says overall—and what it does *not* say

These examples show that pruning has **already been explored in computational pathology**. It would be incorrect to claim that model pruning itself is novel in this field. There is published work on unstructured weight pruning, structured CNN channel pruning, attention-head pruning, foundation-model block/token pruning, and WSI patch selection.

However, comparisons remain difficult because studies differ in architecture, task, dataset, measurement hardware, and what they count as “pruning.” A reported 80% reduction in zero-valued weights is not directly comparable to a 35% FLOP reduction or a 5× wall-clock improvement.

**Potential open questions**, rather than established research gaps, include whether the same physically pruned architecture retains accuracy across many pathology tasks; whether pruning and distillation help one another; whether small but clinically important structures are disproportionately harmed; and how performance changes across new scanners, sites, and patient groups. Specific novelty claims require a focused, up-to-date systematic literature search.

---

## 9. How to read the literature in a sensible order

If the goal is to become technically confident without being overwhelmed:

1. **Han et al. (2015)** — understand ordinary magnitude/weight pruning and why it was introduced.
2. **Li et al. (2017)** — understand structured CNN filter pruning and real computation reduction.
3. **Adnan et al. (2024)** — see those ideas applied to real pathology classification and nuclei segmentation.
4. **Michel et al. (2019)** and **DynamicViT (2021)** — learn why attention heads and tokens are different pruning targets.
5. **TAP-Path (2026)** — examine direct compression of a large pathology foundation model.
6. **DTC-WSI (2026)** and **EAGLE (2026)** — understand why reducing the number of tiles can be as important as reducing the model itself.

For each paper, write down five answers: **what is removed; how importance is calculated; when retraining occurs; what was measured on real hardware; which dataset and task were used**.

---

## 10. Applying pruning to the tile-and-error / CS407 pipeline

### 10.1 Match the actual project plan

The [repository's plan](https://github.com/tile-and-error/imagenet-classification/blob/main/PLAN.md) deliberately starts with a reproducible **ImageNet-style ResNet18 classification baseline**. In Phase 1 the goal is to train/evaluate/benchmark one model; it is *not yet* to build compression methods. It allows tiny synthetic **FakeData** for smoke tests (checking that code runs) and a genuine ImageFolder subset for meaningful classification experiments.

The plan then introduces **knowledge distillation** in Phase 2, and in **Phase 3** explicitly names baseline vs KD vs **pruning** vs **quantisation**. It says to separate **unstructured masks** from **physically removed channels**, and to benchmark actual deployed models after transforms. **Phase 4** expands to segmentation, pathology, and tile selection.

Your **30 October CS407 specification** emphasises efficient pathology foundation models, distillation and quantisation, and evaluating quality alongside inference cost. It does not make pruning a named primary objective, so pruning is best presented as an *additional technique/workstream consistent with the later repository plan*, rather than something the original specification already promised.

### 10.2 Start with two small experiments that teach different lessons

**Experiment A — unstructured magnitude-pruning baseline (educational):**

1. Obtain a trained ResNet18 checkpoint and an honest held-out validation set.
2. Record its unpruned top-1/top-5 accuracy, parameters, dense checkpoint size, latency and throughput.
3. Apply global magnitude pruning at several rates, for example 20%, 40%, 60% and 80%.
4. Measure accuracy **before** and **after** optional fine-tuning, keeping the same evaluation protocol.
5. Report the fraction of zero-valued weights separately from physical parameter count and runtime.

**Main question:** *How much functional redundancy is present, and does sparse masking alone help the speed that users actually experience?*

**Experiment B — physical channel pruning (efficiency-focused):**

1. Start from the same original ResNet18 checkpoint.
2. Rank eligible convolutional filters using L1 or L2 norms.
3. Prune modest channel fractions, taking account of ResNet's residual dependencies.
4. **Physically rebuild dependent layers** and verify that a forward pass still works.
5. Fine-tune; export a standalone compressed inference artifact; measure accuracy, dense parameter count and actual latency.
6. Compare at several pruning budgets, plus a random structured-pruning baseline.

**Main question:** *Which physically smaller model provides the best quality–cost trade-off on the target hardware?*

Do **not** use `FakeData` to make accuracy claims. Also avoid presenting a model trained on a small ImageNet subset as if its score were directly comparable with a standard ImageNet-1k pretrained benchmark.

### 10.3 Why pruning is a **ModelTransform**—but not always *only* that

The repo plan uses several extension points:

- **`ModelTransform`:** a transformation of an `nn.Module` (PyTorch's base class for trainable models). Appropriate for one-shot pruning or physically rewriting channels after a trained checkpoint is loaded.
- **`Callback`:** an operation invoked during training at specified points. Appropriate for scheduled/gradual pruning while the optimiser is running.
- **`LossBinding`:** a named connection that supplies model outputs and targets to a loss calculation. Appropriate for adding a sparsity penalty or KD loss.
- **Graph node:** a computation with named inputs/outputs, such as a learned dynamic token selector inserted into the model graph.
- **`DataTransform`:** a procedure that changes input data, potentially for deterministic tissue-tile subsampling before the expensive model.
- **`Stage`:** a reproducible experiment step that produces a result and artifact, such as `PruneStage`, fine-tuning, evaluation, or export.

The **graph** is the set of computations and data connections making up an experiment; a **node** is one component in it. An **artifact** is a saved output—such as a model checkpoint—that future steps can load. A **recipe** is the ordered sequence of experimental stages.

A sensible experimental sequence is:

```text
Load or train the original model
        ↓
Evaluate + benchmark original (save exact artifact)
        ↓
Apply pruning ModelTransform
        ↓
Evaluate immediately: how much accuracy was lost?
        ↓
Fine-tune (optionally using KD)
        ↓
Evaluate + benchmark physically pruned model
        ↓
Optional quantisation (on a supported backend)
        ↓
Evaluate + benchmark again
```

**Keep your original model immutable.** Every pruning stage should save a new model with a clear link to its parent checkpoint and include the pruning method, target/achieved sparsity, ranking data, fine-tuning conditions, and hardware benchmark configuration.

### 10.4 PyTorch note: a mask is not a physically smaller network

PyTorch provides `torch.nn.utils.prune` for ordinary weight masking. For example, `prune.l1_unstructured(layer, name="weight", amount=0.3)` makes 30% of a layer's selected weight entries zero through pruning reparameterisation. **`prune.remove(layer, "weight")` removes the masking machinery, but it does not shrink the tensor's shape.**

For physical structural pruning, dependency-aware tools such as [Torch-Pruning / DepGraph](https://github.com/VainF/Torch-Pruning) can identify connected layers that need coordinated surgery. Treat library APIs and model support as version-specific: validate them on your installed PyTorch/torchvision versions. Run tests that check actual tensor shapes, module outputs, saved/reloaded predictions, and parameter counts.

**Implementation caution:** filter removal can invalidate an existing optimizer's parameter references and state. After physical surgery, construct a fresh compatible optimizer for fine-tuning rather than assuming old optimizer state remains valid.

### 10.5 The first pathology follow-on experiment

Do **not** jump immediately from ResNet18/ImageNet to structural surgery on an enormous ViT. A more interpretable progression is:

**Step 1 — tile classification:** take a small, clearly defined histology tile-classification task (for example, colorectal tissue-type classification). Compare unpruned CNN, unstructured magnitude-pruned CNN, and physically channel-pruned CNN. This keeps the problem similar to ImageNet, changing primarily the visual domain.

**Step 2 — pathology pretrained encoder:** choose a supported pretrained tile model and examine whether removing CNN channels or Transformer blocks/heads retains useful representations. If features are reused across tasks, compare **frozen feature extraction** (weights do not change, only embeddings are extracted) and task-specific fine-tuning.

**Step 3 — slide-level MIL classification:** extract tile embeddings, build a simple slide-level aggregator, and compare retaining all tile features versus selecting/merging some. Benchmark separately (a) tile extraction and (b) slide aggregation, then measure the full pipeline.

**Step 4 — optional segmentation:** only after the framework can handle pixel masks and spatially structured outputs, try nuclei or tissue segmentation. Pruning here must preserve detailed boundaries and individual object counts, not merely slide-level labels.

### 10.6 Experiments to make pruning a research contribution

A compact, defensible **ablation study**—an experiment removing or changing one technique at a time—could compare:

| Run | Training | Pruning | Quantisation | What it tests |
|---|---|---|---|---|
| A | Standard supervised | None | None | Reference baseline |
| B | Standard supervised | Unstructured magnitude | None | Sparsity / redundancy only |
| C | Standard supervised | Physical channel/filter | None | Real structural speedup |
| D | Distillation from a teacher | None | None | Benefit of teacher guidance alone |
| E | Distillation + post-prune KD fine-tuning | Physical channel/filter | None | Whether KD helps accuracy recovery |
| F | Same controlled setup as C or E | Physical channel/filter | Supported PTQ | Whether a second compression step compounds savings |

**PTQ (post-training quantisation)** means converting supported weights and/or activations to lower numerical precision after training, possibly using a calibration set. **Quantisation** reduces bits per number; it is a different operation from pruning. Some pruning and quantisation orders may behave differently, so report the order explicitly and only use a supported hardware/software backend.

Optional new questions that are actually testable:

- Do L1, L2, and activation-based structured pruning select the same channels on natural images versus histology tiles?
- Is task-specific fine-tuning enough to recover accuracy, or does KD from an unpruned teacher help more?
- Does a model pruned for one tissue task retain transferable embeddings for a different tissue task?
- Do class-wise rare-disease scores degrade earlier than average accuracy when pruning increases?
- Does tile selection give greater total WSI savings than model pruning at the same quality level—and does combining them still help?

These are **candidate hypotheses**, not assertions that no one has studied them.

---

## 11. Experimental discipline: mistakes that can undermine the results

**1. Comparing different models without controlling training.** Use the same splits, preprocessing, optimiser family, relevant training budget, seeds and evaluation protocol where possible. When fine-tuning time differs, say so.

**2. Reporting only a sparsity percentage.** Always give actual remaining parameters, tensor shapes, precision, FLOPs (with methodology), checkpoint bytes and measured runtime.

**3. Calling a mask a speedup.** Explicitly label masked/unstructured runs and physically compressed runs separately. Check if sparse kernels were actually used.

**4. Measuring only the model's forward pass for WSI inference.** If the user needs a slide prediction, report tile-reading, tissue masking, selection, feature extraction, aggregation and any post-processing.

**5. Applying pruning to the wrong deployment artifact.** The teacher, training-only projectors, caches, optimizer state, or extra heads may exist during training but should not be included when measuring the student intended for deployment.

**6. Choosing pruning settings using the test set.** Select rates and methods on training/validation data; test once using the locked experimental protocol.

**7. Losing spatial information.** For slide-based methods, retain tile coordinates, magnification/MPP and a way to map retained or discarded tokens back onto the WSI; this makes diagnostic failure inspection possible.

**8. Ignoring external validity.** **External validation** tests on data from institutions, scanners, cohorts or sources distinct from training. It is more informative for real deployment than another random split from the same hospital. The project specification notes possible TCGA pretraining overlap, so dataset provenance needs particular care.

**9. Ignoring the hardest classes.** Use per-class sensitivity, balanced accuracy (average recall across classes), and macro-F1 (average F1 across classes). **F1** combines precision (fraction of predicted positives that are correct) and recall. For segmentation, report Dice, intersection-over-union (IoU) and, for instance tasks, PQ or AJI as appropriate.

**10. Claiming that reduced FLOPs prove clinical readiness.** Clinical use additionally requires rigorous validation, safety analysis, regulatory considerations and integration with healthcare workflows. A university research result is not a diagnostic device.

---

## 12. A suggested minimum viable pruning workstream

**Milestone A — learning + correct baseline.** Read Han 2015, Li 2017 and Adnan 2024; ensure the repository's Phase 1 baseline runs and reloads consistently. Implement or reuse accurate latency and parameter-count measurement.

**Milestone B — sparse-weight demonstration.** Implement global and layer-wise magnitude masks, compare with random masks, and explicitly show why dense runtime may barely change. Save all artefacts and metrics.

**Milestone C — first actual speedup.** Implement physical ResNet18 channel pruning, with a dependency-aware tool if appropriate. Compare one-shot with iterative pruning, with and without fine-tuning. Test repeatability and exported checkpoint loading.

**Milestone D — pathology transfer.** Run the same method on a pathology tile task; compare class-wise performance and external robustness. Ensure the dataset license and patient-level separation are correct.

**Milestone E — combine techniques.** Explore KD-assisted recovery and supported quantisation only after the pruning-only baselines are trustworthy.

**Milestone F — research extension.** Depending on time and access, test a pathology ViT block/head-pruning method or a slide-token-selection method. Keep **model pruning** and **tile selection** as separately reported interventions.

The strongest initial success criterion is *not* “reach 90% sparsity.” It is: **produce a smaller, reloadable, demonstrably faster model, quantify the predictive cost, and explain exactly what was removed.**

---

## 13. Glossary of recurring terms

| Term | Meaning |
|---|---|
| **Ablation study** | Controlled experiment changing one component to learn whether it helps. |
| **Activation** | Intermediate numerical output produced when a network processes an input. |
| **Attention** | A mechanism for weighting information from other tokens or regions. |
| **Attention head** | One parallel attention calculation in a multi-head attention module. |
| **Backbone / encoder** | Main network that transforms an image into useful features. |
| **Batch size** | Number of inputs processed together in one model call or training step. |
| **Calibration set** | Representative examples used to characterise behaviour or numerical ranges without necessarily training weights. |
| **Channel / feature map** | One output dimension/grid created by a CNN filter. |
| **Checkpoint** | Saved model weights, optionally including training state. |
| **Class imbalance** | Some labels occur much more often than others. |
| **CNN** | Convolutional neural network, built around sliding learnable filters. |
| **Distillation (KD)** | Training a student to reproduce useful behaviour of a teacher. |
| **Domain shift** | Differences between training data and later evaluation/deployment data. |
| **Embedding** | Vector representing learned features of an image or tile. |
| **Fine-tuning** | Continuing training from pretrained weights, often for a specific task or after pruning. |
| **FLOPs** | Estimated count of floating-point arithmetic operations. |
| **Gradient** | Sensitivity of a loss to a small parameter change. |
| **H&E** | Haematoxylin and eosin tissue stain. |
| **Inference** | Using a trained model to make a prediction. |
| **Instance segmentation** | Assigning individual identities/regions to separate objects, such as cell nuclei. |
| **IoU / Dice** | Scores measuring overlap between predicted and labelled regions. |
| **Keep rate** | Fraction of units or tokens retained after pruning. |
| **Latency** | Elapsed time required for an inference request. |
| **Logits** | Raw prediction scores before probability normalisation. |
| **Magnification / MPP** | Imaging scale; microns per pixel is physical length represented by a pixel. |
| **Mask** | Binary switches that suppress weights or units without necessarily changing tensor dimensions. |
| **MIL** | Multiple-instance learning: learning from bags of instances, such as tiles in one labelled slide. |
| **ModelTransform** | Repository-plan extension point that applies a modification to a PyTorch model. |
| **N:M sparsity** | Keep N weights in every specified group of M weights. |
| **Parameter** | A learned number (weight or bias) within a model. |
| **Pareto frontier** | Set of efficient trade-offs where no model improves both quality and cost simultaneously. |
| **Patch / tile** | Small image region. In ViTs, a patch often becomes a token; in pathology, a tile may be separately encoded before slide aggregation. |
| **Post-training pruning** | Pruning performed after a model has been trained. |
| **Pruning rate / sparsity** | Fraction of chosen weights, groups or tokens removed or disabled. |
| **Quantisation** | Lowering the number of bits used to represent numerical values. |
| **Residual / skip connection** | Path that carries information around intermediate layers; introduces pruning shape dependencies. |
| **Self-supervised learning** | Representation learning from signals derived from data, without requiring manual labels for every training example. |
| **Semantic segmentation** | Assigning a class to each pixel, without necessarily separating distinct objects of the same class. |
| **Structured pruning** | Removing complete computational groups such as channels, heads or layers. |
| **Throughput** | Number of inputs completed per unit time. |
| **Token** | Vector representing a patch or other unit processed by a Transformer. |
| **U-Net / encoder–decoder** | Architecture that compresses image representations then reconstructs spatial predictions, often with skip connections. |
| **Unstructured pruning** | Zeroing arbitrary individual weights, often without shrinking tensor dimensions. |
| **ViT** | Vision Transformer, a Transformer used for images. |
| **Weak supervision** | Training from coarse or incomplete labels, such as a slide diagnosis without patch labels. |
| **WSI** | Whole-slide image: high-resolution digital scan of a pathology slide. |

---

## 14. Clickable reference library

The following papers are grouped by what you can learn from them. Descriptions and quantitative claims in this guide are based on the linked papers, project plan, and project specification; the research papers' reported numbers are not independent benchmark reproductions.

### Core pruning foundations

1. **LeCun, Denker & Solla (1989).** [*Optimal Brain Damage*](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html) — early sensitivity-based pruning.
2. **Hassibi & Stork (1992).** [*Second Order Derivatives for Network Pruning: Optimal Brain Surgeon*](https://papers.neurips.cc/paper_files/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html) — second-order loss and weight interactions.
3. **Han, Pool, Tran & Dally (2015).** [*Learning both Weights and Connections for Efficient Neural Networks*](https://arxiv.org/abs/1506.02626) — foundational magnitude/connection pruning.
4. **Li et al. (2017).** [*Pruning Filters for Efficient ConvNets*](https://arxiv.org/abs/1608.08710) — simple structural filter pruning.
5. **Liu et al. (2017).** [*Learning Efficient Convolutional Networks Through Network Slimming*](https://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) — batch-normalisation-based channel sparsity.
6. **Molchanov et al. (2017).** [*Pruning Convolutional Neural Networks for Resource Efficient Inference*](https://research.nvidia.com/publication/2017-04_pruning-convolutional-neural-networks-resource-efficient-inference) — Taylor importance.
7. **Zhu & Gupta (2017).** [*To Prune, or Not to Prune*](https://arxiv.org/abs/1710.01878) — pruning schedules and compression.
8. **Louizos, Welling & Kingma (2018).** [*Learning Sparse Neural Networks through L0 Regularization*](https://arxiv.org/abs/1712.01312) — learned sparse gates.
9. **Frankle & Carbin (2019).** [*The Lottery Ticket Hypothesis*](https://dspace.mit.edu/entities/publication/d2335291-d7a7-418b-8708-0f47209827d6) — sparse trainable subnetworks.
10. **Gale, Elsen & Hooker (2019).** [*The State of Sparsity in Deep Neural Networks*](https://research.google/pubs/the-state-of-sparsity-in-deep-neural-networks/) — large-scale sparsity study.
11. **Michel, Levy & Neubig (2019).** [*Are Sixteen Heads Really Better than One?*](https://proceedings.neurips.cc/paper/2019/hash/2c601ad9d2ff9bc8b282670cdd54f69f-Abstract.html) — Transformer head redundancy.
12. **Rao et al. (2021).** [*DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification*](https://proceedings.neurips.cc/paper_files/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html) — adaptive token removal.
13. **Yin et al. (2022).** [*A-ViT: Adaptive Tokens for Efficient Vision Transformer*](https://openaccess.thecvf.com/content/CVPR2022/html/Yin_A-ViT_Adaptive_Tokens_for_Efficient_Vision_Transformer_CVPR_2022_paper.html) — token-adaptive computation.
14. **Fang et al. (2023).** [*DepGraph: Towards Any Structural Pruning*](https://github.com/VainF/Torch-Pruning) — coordinated removal of shape-dependent layers.

**Further variants:** [SNIP (Lee et al., 2019)](https://arxiv.org/abs/1810.02340), [GraSP (Wang et al., 2020)](https://arxiv.org/abs/2002.07376), and [Movement Pruning (Sanh et al., 2020)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/eae15aabaa768ae4a5993a8a4f4fa6e4-Abstract.html) cover pruning before training and during pretrained-model adaptation.

### Computational pathology pruning and related reduction

15. **[2019].** [*Breast cancer histopathology image classification through assembling multiple compact CNNs*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6805574/) — SE-guided channel pruning.
16. **[2022].** [*Deep Neural Network Pruning for Nuclei Instance Segmentation in H&E-Stained Histological Images*](https://arxiv.org/abs/2206.07422) — unstructured magnitude pruning in nuclear segmentation.
17. **Adnan et al. (2024).** [*Structured Model Pruning for Efficient Inference in Computational Pathology*](https://arxiv.org/abs/2404.08831) — filter pruning of HoVer-Net and ResNet18.
18. **[2024].** [*Creating an atlas of normal tissue for pruning WSI patching through anomaly detection*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10873359/) — discarding redundant normal tissue patches.
19. **[2025].** [*Vision Transformers for Histopathological Image Classification with Efficient Head Pruning*](https://doi.org/10.1016/j.procs.2025.09.348) — attention-head pruning and KD.
20. **[2026].** [*A deep learning framework for efficient pathology image analysis* (EAGLE)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13324285/) — guided selection before expensive tile encoding.
21. **Li et al. (2026).** [*Whole slide image redundancy reduction for efficient pathological diagnosis* (WSIR²)](https://doi.org/10.1016/j.bspc.2025.109289) — WSI token selection and merging.
22. **Rahman et al. (MIDL 2026).** [*DTC-WSI: Dynamic Token Compression for Whole-Slide Images*](https://proceedings.mlr.press/v315/rahman26b.html) — progressive slide-token compression.
23. **Hasan, Yeafi & Islam (2026 preprint).** [*TAP-Path*](https://arxiv.org/abs/2609.04071) — block and token pruning of the Virchow2 pathology encoder.
24. **Bozkurt et al. (2026 preprint).** [*SLICEChat*](https://arxiv.org/abs/2609.24894) — region-aware token pruning in WSI-language models.

### Tools, project context, and background

25. [PyTorch pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial) — masks, weight reparameterisation, PyTorch APIs.
26. [Torch-Pruning / DependencyGraph](https://github.com/VainF/Torch-Pruning) — structural pruning and shape dependencies; check the installed version's docs.
27. [tile-and-error/imagenet-classification — `PLAN.md`](https://github.com/tile-and-error/imagenet-classification/blob/main/PLAN.md) — Phase 1 baseline, Phase 2 KD, Phase 3 compression, Phase 4 pathology.
28. [tile-and-error/imagenet-classification — PR #4](https://github.com/tile-and-error/imagenet-classification/pull/4) — scaffolded building blocks; the plan is a design intention, not proof every phase has been implemented.
29. **Chen et al. (2024).** [*Towards a general-purpose foundation model for computational pathology*](https://pmc.ncbi.nlm.nih.gov/articles/PMC11403354/) — UNI and the motivation for transferable pathology image representations.
30. **Lu et al. (2021).** [*Data-efficient and weakly supervised computational pathology on whole-slide images*](https://pmc.ncbi.nlm.nih.gov/articles/8711640/) — CLAM and slide-level MIL context.
31. **CS407 team specification (30 October 2026).** *Lightweight Foundation Models for Whole-Slide Image Representation Learning* — user-supplied project document; especially sections 1–3 on objectives, datasets, evaluation and libraries.

**Final perspective:** start by learning how to remove **weights**, then how to remove **entire computations**, and only then how to remove **parts of an enormous pathology image**. The most convincing research result will combine a clearly defined removal method with preserved predictive quality and an actual, reproducible inference-cost improvement.
