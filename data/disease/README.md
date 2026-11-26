# Crop Disease Dataset

The application expects a trained model at `artifacts/crop_disease_model.pt`.
The original Kaggle dataset cited in the project notes is not used because its
files are marked “Data files © Original Authors.” Instead, the preparation
script uses datasets with declared licenses that permit model training and
records attribution here. Downloaded and derived images are stored in the
ignored `source/` directory; class-balanced train/validation/test splits go
under `processed/`. The images and trained model are not committed to Git.

| Crop | Source | License and coverage |
|---|---|---|
| Rice | [Project-AgML rice leaf bacterial/fungal dataset](https://huggingface.co/datasets/Project-AgML/rice_leaf_bacterial_fungal) | CC BY 4.0; raw set includes healthy leaves, diseases, and rice hispa. Original dataset: Hasan, M. (2023), [Mendeley Data](https://doi.org/10.17632/HX6F852HW4.2). |
| Maize (corn) | [Project-AgML maize disease classification](https://huggingface.co/datasets/Project-AgML/maize_disease_classification) | CC BY 4.0; healthy, MLN, and MSV classes. Original dataset: Mduma & Mayo (2023), [Mendeley Data](https://doi.org/10.17632/fkw49mz3xs.1). |
| Soybean | [Project-AgML soybean leaf disease classification](https://huggingface.co/datasets/Project-AgML/soybean_leaf_disease_classification) | CC BY 4.0; healthy and disease classes. Original dataset: Kotwal & Kashyap (2023), [Mendeley Data](https://doi.org/10.17632/bshkvgbzpt.1). The source's `Root_images` class is excluded because this model classifies leaf images. |
| Wheat | [Focused Rust Disease Images (FRDI)](https://doi.org/10.5281/zenodo.24455348) | CC BY 4.0; the training archive contains 85 paired field images and masks. **Experimental only:** wheat support is limited to rust/healthy patches derived from masks and green-leaf pixels; this is not a general wheat disease classifier. |

Attribution: dataset images are provided by their respective creators and are
used under the linked licenses. The preparation step resizes images, combines
class names with crop names, makes deterministic train/validation/test splits,
and derives wheat rust/healthy patches from the published masks. Source-image
groups stay together across all splits to prevent patch leakage. Review each
source's license and attribution requirements before redistributing derived
data or model artifacts.

The preparation script uses the Hugging Face dataset viewer API for rice,
maize, and soybean rather than downloading their full multi-GB archives. It
downloads the wheat training archive to derive experimental rust/healthy
patches. The project already declares the required Pillow and NumPy
dependencies. By default, it fetches up to 100 images per class. Run it from
the repository root:

```powershell
python scripts/prepare_disease_dataset.py
```

Use `--max-per-class N` to change the per-class cap. Use `--force` to rebuild
the generated `processed/` splits after changing options.

Train and evaluate the classifier:

```powershell
python scripts/train_disease_model.py --data-dir data/disease/processed
```

The prepared ImageFolder layout is:

```text
data/disease/processed/
├── train/
│   ├── Rice___Healthy/
│   ├── Maize___MLN/
│   └── ...
├── val/
│   └── <same disease class names>
└── test/
    └── <same disease class names>
```

Each class name includes the crop, for example `Rice___Healthy` and
`Maize___MLN`, so the API can return the crop and disease separately. Wheat
labels are limited to `Wheat___Healthy` and `Wheat___Rust` and must not be
interpreted as a comprehensive diagnosis.
