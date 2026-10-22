# park-model

Detecting Parkinson's disease from DaTscan (dopamine transporter SPECT) images with classical machine learning.

DaTscan images from the PD / non-PD benchmark dataset of Tagaris et al. (2018) are resized to 150×150, flattened, and used to train three classifiers. Each model's hyperparameters are tuned with 5-fold `GridSearchCV`, and the models are then compared on accuracy and training time (paired t-tests).

## What's here

| File | What it does |
|---|---|
| `park_model_svm.py` | Support vector machine (`SVC`), grid-searched, saved to `svm_model.joblib` |
| `park_model_dtree.py` | Decision tree, `max_depth` searched over 1–30, saved to `decision_tree_model.joblib` |
| `park_model_lr.py` | Logistic regression, grid-searched, saved to `logistic_regression_model.joblib` |
| `t_test.py` | Paired t-tests comparing training times across the three models |
| `alg_acc.py` | Bar chart of average training time per algorithm |
| `GUI.py` | Tkinter app: load a DaTscan image, pick a model, get a PD / non-PD prediction with class probabilities |

## Run it

```bash
pip install scikit-learn scikit-image pandas numpy matplotlib scipy joblib pillow
```

1. Put the PNG images in `pd_image_data/PD/` and `pd_image_data/NON_PD/`, and set `datadir` in each training script to that folder.
2. Train a model, e.g. `python park_model_svm.py` (writes the `.joblib` file).
3. Run `python GUI.py` to classify individual images.

## Dataset

```bibtex
@article{tagaris2018machine,
  title={Machine Learning for Neurodegenerative Disorder Diagnosis—Survey of Practices and Launch of Benchmark Dataset},
  author={Tagaris, Athanasios and Kollias, Dimitrios and Stafylopatis, Andreas and Tagaris, Georgios and Kollias, Stefanos},
  journal={International Journal on Artificial Intelligence Tools},
  volume={27},
  number={03},
  pages={1850011},
  year={2018},
  publisher={World Scientific}
}
```
