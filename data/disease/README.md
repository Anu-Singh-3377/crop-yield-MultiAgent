# Crop Disease Dataset Integration

This directory is intentionally empty. The notebook references the Kaggle dataset:

- Dataset title: New Plant Diseases Dataset
- Dataset slug: `vipoooool/new-plant-diseases-dataset`
- Kaggle dataset ID: `78313`
- Kaggle source/datasetVersion ID recorded in the notebook: `182633`
- Kaggle current dataset version at verification time: version `2`
- License shown by Kaggle: `Data files © Original Authors`

Do not download or redistribute this dataset automatically. Its Kaggle license is not
an SPDX-style open-source license, so confirm permissions for your intended use.

Expected `ImageFolder` layout:

```text
data/disease/
├── train/
│   ├── Tomato___Healthy/
│   ├── Tomato___Early_blight/
│   └── ...
├── val/
│   └── <same disease class names>
└── test/
    └── <same disease class names>
```

Each class directory must contain plant leaf images. Training is not performed until
an approved local dataset is supplied.
