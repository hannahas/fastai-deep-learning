# Lesson 1 — Image Classification

## What we built
A three-class image classifier that distinguishes between:
- Astronomy images (nebulae)
- Microscopy cell images
- Pancakes

~96 training images total (~30 per class), trained in under a minute on Kaggle GPU.

---

## The model: Convolutional Neural Network (CNN)

We're using a **ResNet18** — a specific type of Convolutional Neural Network (CNN).
CNNs are designed for image data. Unlike a standard feedforward network that 
treats every pixel independently, a CNN uses convolutional layers that scan 
across the image looking for local patterns — edges, curves, textures — and 
build up increasingly complex features in deeper layers.

ResNet18 specifically has 18 layers and uses "residual connections" that pass 
information directly between non-adjacent layers, which helps with training 
stability in deeper networks.

---

## Transfer learning

ResNet18 was pretrained on **ImageNet** — 1.2 million images across 1000 categories.
During that training it learned to recognize universal visual features:
- Early layers: edges, color gradients, simple textures
- Middle layers: shapes, patterns, object parts
- Later layers: complex object features

These learned parameters (weights) are reused as a starting point for our 
classifier. This is called **transfer learning** — knowledge from one task 
transfers to another.

The final layer of ResNet18 (originally sized for 1000 ImageNet classes) is 
replaced with a new layer sized for our 3 classes.

---

## Fine-tuning with `fine_tune(3)`

Training happens in two phases:

**Phase 1 (1 epoch):** All pretrained layers are frozen. Only the new final 
layer is trained. This gets the new layer to a reasonable starting point quickly.

**Phase 2 (3 epochs):** The entire network is unfrozen. All layers are updated,
but the pretrained layers use a much smaller learning rate so they change only 
slightly. The final layer continues getting larger updates.

Result: the pretrained layers get nudged toward our domain (nebulae, cells, 
pancakes) while retaining their general visual knowledge.

---

## Would training from scratch be better?

No — for two reasons:

1. **Insufficient data.** Training ResNet18 from scratch requires tens of 
thousands of images per class. With ~30 images per class, a randomly 
initialized network has millions of parameters to fit and almost no data 
to fit them to. It would massively overfit.

2. **Pretrained features are universally useful.** Edges, textures, curves, 
and color patterns appear in all images — nebulae, cells, and pancakes alike. 
There's no reason to relearn these from scratch.

The only case where training from scratch might win is if your data is so 
unlike natural photographs that pretrained features are actively misleading 
(e.g. raw radar data, genomic heatmaps). Even then, transfer learning 
usually still helps.

---

## Key concepts

**DataBlock** — fastai's data pipeline builder. Defines what the data is, 
where to find it, how to split train/validation, how to get labels, and 
what transforms to apply. Separates pipeline definition from execution.

**vision_learner** — combines data, model, and metrics into a single object. 
Automatically downloads pretrained weights and replaces the final layer for 
your number of classes.

**error_rate** — the metric displayed during training. Just 1 - accuracy. 
Doesn't affect training, just for monitoring.

**Confusion matrix** — shows which classes the model confuses. Our model 
confused one microscopy cell image as a pancake — both have rounded repeating 
patterns. The model sees texture, not meaning.

**plot_top_losses** — shows images the model was most wrong or least confident 
about. Useful for understanding failure modes and improving training data.

---

## Results

| Metric | Value |
|--------|-------|
| Training images | ~96 |
| Epochs | 4 (1 frozen + 3 fine-tuned) |
| Final error rate | 5.3% |
| Final accuracy | 94.7% |

One microscopy cell image misclassified as pancakes — visually similar 
rounded texture pattern.

## Key concepts
- DataBlock API for loading image data
- Transfer learning with pretrained ResNet
- fine_tune() vs fit_one_cycle()
- Confusion matrix for evaluating classifier

## Questions / things to look up
- What is the difference between fine_tune() and fit_one_cycle()?
- How does fastai decide what pretrained model to use?

## Code snippets worth remembering

```python
# Standard image classification setup
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```