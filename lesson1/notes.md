# Lesson 1 — Image Classification

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