# Master_Thesis

Master Thesis: Image Segmentation on Arcade Dataset
This repository contains the code and resources for my Master’s Thesis on performing image segmentation on the Arcade Dataset. The primary focus is to evaluate the performance of three segmentation models—UNet, SAM2, and MedSAM—by implementing and testing them within a unified PyTorch Lightning framework.

Project Overview
The objective of this project is to develop and evaluate various deep learning models for image segmentation, with a specific focus on applications in fine-grained segmentation tasks. Each model is evaluated on segmentation accuracy metrics such as Dice Score and Binary Cross-Entropy (BCE) Loss.

Models Implemented
UNet: A traditional encoder-decoder architecture commonly used for biomedical image segmentation.
SAM2: An architecture from the Segment Anything Model (SAM) series, optimized for general segmentation tasks.
MedSAM: A version of SAM specifically designed for medical image segmentation, leveraging specialized configuration for improved performance on medical images.

Prerequisites
Ensure you have Python 3.x installed. Recommended libraries include:
torch and torchvision
pytorch_lightning
tensorboard
yaml
matplotlib
PIL
