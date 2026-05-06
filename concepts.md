# Deep Learning Concepts

A running glossary of concepts encountered in the fast.ai course.

---

## DataBlock
fastai's API for defining how to load and prepare data. Specifies:
- `blocks` — input and output types (e.g. ImageBlock, CategoryBlock)
- `get_items` — how to find data items (e.g. get_image_files)
- `splitter` — how to split into train/validation sets
- `get_y` — how to get labels (e.g. parent_label uses folder names)
- `item_tfms` — transforms applied to each item (e.g. Resize)

## Transfer Learning
Using a model pretrained on a large dataset (e.g. ImageNet) as a starting 
point for a new task. The pretrained weights encode general features 
(edges, textures, shapes) that transfer to new domains. Fine-tuning adjusts 
these weights for the specific task.

## parent_label
fastai built-in that extracts a label from the parent folder name of a file. 
Assumes data is organized into folders by class (e.g. `cats/img001.jpg` → label `cats`).

## Resize (item_tfms)
Standardizes image dimensions so they can be batched. `method='squish'` 
stretches images to the target size, preserving all pixels but distorting 
aspect ratio. Alternatives: `crop` (cuts edges), `pad` (adds blank space).