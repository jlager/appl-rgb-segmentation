# APPL RGB Segmentation

This repository contains code and resources for training and evaluating fast, accurate plant segmentation models using RGB imagery from the Advanced Plant Phenotyping Laboratory (APPL) at ORNL. The goal is to develop a robust, generalizable segmentation model capable of handling diverse species, views, and conditions across the facility.

We compare classical convolutional architectures (U-Net from `segmentation_models.pytorch`) with fine-tuned transformer-based models (ViT backbones from `timm`) for semantic segmentation of plants. The dataset includes images of poplar and switchgrass plants captured from top and side views under controlled conditions.