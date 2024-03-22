# Skoltech Machine Learning Course Project 2024

Welcome to our Machine Learning course project repository! This project is part of the 2024 curriculum at Skoltech, aimed at exploring innovative solutions and advancing the field of machine learning.

## Project Overview

This project is focused on creating an advanced approach for the accurate segmentation and classification of brain tumors, utilizing a dataset called “Brain MRI segmentation” for segmentation task and then train classification on  "Brain Tumor Classification" dataset that are filled with magnetic resonance (MR) images of the brain. Our goal is to accurately segment and classify various types of tumors through the application of various deep learning models, including CNN (Convolutional Neural Networks), U-Net, ResNet, and others. This endeavor seeks to leverage the capabilities of these models to improve the precision and efficiency of brain tumor diagnosis.

## Team Members

| Name              | Role                | Contact Information |
|-------------------|---------------------|---------------------|
| Hasaan Maqsood    | Project Lead        | [Email](Hasaan.Maqsood@skoltech.ru) |
| Inna Larina       | CV Engineer         | [Email](inna.larina@skoltech.ru) |
| Iana Kulichenko   | Data Scientist      | [Email](Iana.Kulichenko@skoltech.ru) |

## Table of Contents

- [Project Overview](#project-overview)
- [Team Members](#team-members)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#Team-Member)
- [License](#license)


# Project File Structure

This repository supports classification and segmentation machine learning tasks, alongside providing presentation and report documentation. Below is a detailed overview of the repository's structure.

## Repository Structure

```bash
├── classification
│ ├── classification.ipynb          # Jupyter notebook with the model training code for classification.
│ └── classification dataset         # Directory containing the dataset for classification tasks.
│    ├── Training                    # Training dataset for the classification model.
│    └── Testing                     # Testing dataset for the classification model.
├── segmentation
│ ├── segmentation.ipynb            # Jupyter notebook with the model training code for segmentation.
│ ├── segm_model.pth                # Saved model weights for the segmentation model.
│ └── segmentation dataset          # Directory containing the dataset for segmentation tasks.
├── Presentation                     # Directory or file with presentation materials.
├── Report                           # Directory or file with detailed project report.
├── Readme.md                        # Overview and guide for using this repository.
└── requirements.txt                 # Required libraries to run the notebooks.

```
Provide step-by-step instructions on how to install the project.
## Installation

```bash
git clone [https://your.project.repo.link.here](https://github.com/Hasaanmaqsood/Skoltech_Machine_learning-2024.git)
cd Skoltech_Machine_learning-2024
pip install -r requirements.txt

```
# Datasets 
## How to download Segmentation Dataset
```bash
# Downloading dataset from kaggle
#upload kaggle json
from google.colab import files
uploaded = files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p /content
!unzip /content/lgg-mri-segmentation.zip -d /content/dataset
```
## How to download Classification Dataset
```bash
!git clone https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet.git
```
## How to load models
Firstly download .pth file from Google Drive (see instuctions Segmentation/Instructions to download weights.txt). Then write code:
```bash
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER = 'efficientnet-b5'
ENCODER_WEIGHTS = "imagenet"
CLASSES = ['Health_brain', 'Tumor']  # Binary classification (background and tumor)
ACTIVATION = 'sigmoid'  # sigmoid activation for binary classification

aux_params = dict(
    pooling='avg',
    dropout=0.3,
    activation=ACTIVATION,
    classes=1,
)

common_params = {
    'encoder_name': ENCODER,
    'encoder_weights': ENCODER_WEIGHTS,
    'classes': 1,
    'activation': ACTIVATION,
    'aux_params': aux_params
}
model = smp.DeepLabV3Plus(**common_params)
model_path = '/content/DeepLabV3+_best.pth'
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model = model.to(DEVICE)
```
## How to make segmentation prediction for your image
```bash
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

img_path = "/content/glioma_tumor.jpg"
img = Image.open(img_path).convert('RGB')
input_image = transform(img)
input_image = input_image.unsqueeze(0).to(DEVICE)  # Add batch dimension and send to device

model.eval()
with torch.no_grad():
    output = model(input_image)
    if isinstance(output, tuple):
        output = output[0]
    prediction = torch.sigmoid(output)
    predicted_mask = (prediction > 0.55).float()


plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# Predicted Mask
plt.subplot(1, 2, 2)
predicted_mask_np = predicted_mask.cpu().squeeze().numpy()
plt.imshow(predicted_mask_np, cmap='gray', vmin=0, vmax=1)
plt.title('Predicted Mask')
plt.axis('off')

plt.show()
```

## Team Member

<a href="https://github.com/Hasaanmaqsood/Skoltech_Machine_learning-2024/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Hasaanmaqsood/Skoltech_Machine_learning-2024"/>
</a>







