# Weekly Project Report

**Project:** 3D Multiclass Brain Tumor Segmentation  
**Reporting period:** October 12–16, 2026  
**Status:** Model training and interactive visualization are implemented; further validation is needed.

> This report is based on the code, model files, and plots currently in the project folder. No dated activity log was available, so it summarizes the current project state rather than attributing individual changes to specific days.

## Overview

The project uses a 3D hybrid U-Net/Transformer model to segment brain-tumor regions from four MRI modalities (T1, T1ce, T2, and FLAIR). The training pipeline targets five labels and includes validation, per-class Dice reporting, checkpoint saving, and early stopping.

## Progress and results

- The training workflow is configured for 50 epochs, a fixed random seed, an 80/20 train-validation split, and a combined cross-entropy plus Dice loss.
- The recorded training-loss plot shows loss decreasing from **2.7412** to **2.1318** over 50 epochs (about **22.2%** lower).
- The recorded validation-loss plot shows loss decreasing from **2.4406** to **2.1365** (about **12.5%** lower); the lowest recorded validation loss is at epoch 50.
- Best-model and resumable-checkpoint files are present.
- Interactive 3D visualizations are available for a tumor surface and for separately colored multiclass regions, with browser-based rotation and zoom.

The available history does not include saved per-class Dice results or an independent test-set evaluation, so segmentation quality beyond the loss trend is not yet established.

## New in this report

- Added a clear distinction between completed implementation and work still requiring validation.
- Added a quantitative summary of the recorded training and validation loss trends.
- Added explicit evaluation and reproducibility tasks to the next-week plan.

## Risks and open items

- Loss reduction alone does not demonstrate strong segmentation performance; per-class Dice scores and visual review against ground-truth masks are needed.
- The available training history is embedded in a plotting script rather than recorded automatically from each run, limiting traceability.
- The current dataset loader expects each patient folder to contain all four MRI modalities and a segmentation file; incomplete or malformed cases may interrupt loading.

## Plan for next week

1. Run inference on held-out cases and record per-class Dice scores.
2. Compare predicted masks with ground truth using representative 2D overlays and the multiclass 3D viewer.
3. Save epoch-level training and validation metrics to a structured log for future comparisons.
4. Verify that dataset inputs are complete and that the final model checkpoint can be loaded for inference.

## Overall assessment

The project has a functioning training pipeline and useful 3D visualization outputs. The next milestone is a reproducible, held-out evaluation that confirms class-level segmentation quality before making stronger performance claims.
