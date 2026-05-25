# Active Contour Segmentation (Snake Algorithm)

![CI](https://github.com/Gescheld/Computer-Vision-Active-Contour/actions/workflows/ci.yml/badge.svg)

> Personal university project — classical computer vision for object segmentation using active contours.

**Status:** Completed (2023) · Personal learning project  
**Context:** Coursework in computer vision; own photographs as test images.

## Overview

This repository implements an **active contour ("snake")** segmentation algorithm in Python. A closed curve of connected points deforms iteratively until it snaps to object boundaries in an image.

The approach minimizes three energy terms:

| Term | Role |
|------|------|
| **Continuity** | Keeps snake points connected |
| **Curvature** | Smooths the contour along edges |
| **Image** | Attracts the snake toward object boundaries |

## What's in this repo

| File | Description |
|------|-------------|
| `Computer Vision - Snake-final.ipynb` | Full implementation and experiments |
| `Tasse.jpeg`, `Tasse2.jpg` | Own test photographs |

## Tech stack

- Python · Jupyter Notebook 6.1.4
- NumPy / image processing (see notebook for dependencies)

## How to run

```bash
git clone https://github.com/Gescheld/Computer-Vision-Active-Contour.git
cd Computer-Vision-Active-Contour
pip install -r requirements.txt jupyter
jupyter notebook "Computer Vision - Snake-final.ipynb"
```

Run all cells. Alternatively, open the notebook in [Google Colab](https://colab.research.google.com/) and upload the image files.

**CI smoke test** (3 snake rounds, no plots):

```bash
pip install -r requirements.txt
python scripts/smoke_test.py
```

## Results

The notebook walks through segmentation on the provided cup images — from initialization to final contour.

## Related work

- [Profile README](https://github.com/Gescheld/Gescheld)
- [LinkedIn](https://www.linkedin.com/in/gesche-held-b49947248/)

## License

No license file specified — contact via LinkedIn if you want to reuse code.
