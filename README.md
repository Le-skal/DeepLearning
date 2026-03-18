<div align="center">

# 🫁 Pneumonia Detection CNN
### *Classification d'Images Medicales par Deep Learning*

<p><em>Detecter la pneumonie sur des radiographies thoraciques avec des reseaux de neurones convolutifs</em></p>

![Status](https://img.shields.io/badge/status-operational-success?style=flat)
![Accuracy](https://img.shields.io/badge/accuracy-89.42%25-blue?style=flat)
![Python](https://img.shields.io/badge/python-3.10+-brightgreen?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deeplearning-pneumonie.streamlit.app)

**[🚀 Tester l'application en ligne](https://deeplearning-pneumonie.streamlit.app)**

<p><em>Built with:</em></p>

![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![TorchVision](https://img.shields.io/badge/TorchVision-0.15-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat&logo=scikit-learn&logoColor=white)

![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-11557c?style=flat)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?style=flat&logo=opencv&logoColor=white)

**Models:**
![ResNet18](https://img.shields.io/badge/ResNet18-Transfer%20Learning-orange?style=flat)
![CNN](https://img.shields.io/badge/CNN-Baseline-blue?style=flat)

---

<img src="outputs/figures/comparison_models.png" alt="Model Comparison" width="600"/>

</div>

## 📋 Objectif

Developper un pipeline CNN capable de distinguer des radiographies thoraciques **normales** de celles presentant une **pneumonie**, avec interpretation visuelle via Grad-CAM.

## 📊 Dataset

**Chest X-Ray Images (Pneumonia)** - [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Set | NORMAL | PNEUMONIA | Total |
|-----|--------|-----------|-------|
| Train | 1,341 | 3,875 | 5,216 |
| Val | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

## 🏗️ Architecture

### CNN Baseline
```
Conv(32) + ReLU + MaxPool
Conv(64) + ReLU + MaxPool
Conv(128) + ReLU + MaxPool
Flatten -> Dense(128) + Dropout -> Dense(1) + Sigmoid
```

### ResNet18 (Transfer Learning)
- ResNet18 pre-entraine sur ImageNet
- Couches convolutives gelees
- Nouvelle tete de classification fine-tunee

## 📈 Resultats

### Comparaison des Modeles

| Metrique | CNN Baseline | ResNet18 | Amelioration |
|----------|--------------|----------|--------------|
| **Accuracy** | 76.28% | **89.42%** | +13.14% |
| **Precision** | 73.91% | **92.19%** | +18.28% |
| **Recall** | 95.90% | 90.77% | -5.13% |
| **Specificite** | 43.59% | **87.18%** | +43.59% |
| **F1-Score** | 83.48% | **91.47%** | +7.99% |

### Visualisations

<div align="center">

| Courbes d'Entrainement | Matrice de Confusion |
|:----------------------:|:--------------------:|
| <img src="outputs/figures/training_curves_resnet18.png" width="400"/> | <img src="outputs/figures/confusion_matrix_resnet18.png" width="350"/> |

| Courbe ROC | Grad-CAM |
|:----------:|:--------:|
| <img src="outputs/figures/roc_curve.png" width="400"/> | <img src="outputs/figures/gradcam_pneumonia.png" width="400"/> |

</div>

## 🔍 Interpretabilite - Grad-CAM

Visualisation des zones utilisees par le modele pour la classification.

- **Zones rouges/jaunes** = regions importantes pour la decision
- Permet de verifier que le modele regarde les bonnes regions (poumons)

## 📁 Structure du Projet

```
DeepLearning/
├── 📂 data/                    # Dataset (non inclus - trop lourd)
├── 📂 samples/                 # Images exemples pour demo
│   ├── NORMAL/
│   └── PNEUMONIA/
├── 📂 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_cnn.ipynb
│   ├── 03_evaluation.ipynb
│   ├── 04_amelioration.ipynb
│   └── 05_gradcam.ipynb
├── 📂 outputs/
│   ├── checkpoints/            # Modeles sauvegardes
│   └── figures/                # Graphiques
├── 📄 app.py                   # Interface Streamlit
├── 📄 requirements.txt
└── 📄 README.md
```

## 🚀 Installation

```bash
# Cloner le repo
git clone https://github.com/Le-skal/DeepLearning.git
cd DeepLearning

# Installer les dependances
pip install -r requirements.txt

# Telecharger le dataset (optionnel)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/ --unzip
```

## 💻 Usage

### Lancer l'interface Streamlit
```bash
streamlit run app.py
```

### Entrainement
Executer les notebooks dans l'ordre :
1. `01_data_exploration.ipynb` - Exploration des donnees
2. `02_training_cnn.ipynb` - Entrainement CNN Baseline
3. `03_evaluation.ipynb` - Evaluation detaillee
4. `04_amelioration.ipynb` - Transfer Learning ResNet18
5. `05_gradcam.ipynb` - Interpretabilite

## ⚠️ Avertissement

> **Cet outil est a but educatif uniquement.**
> Il ne remplace pas un diagnostic medical professionnel.

## 🎓 Conclusion

- Le **Transfer Learning avec ResNet18** ameliore significativement les performances (+13% accuracy)
- La **specificite** passe de 44% a 87% (meilleure detection des cas normaux)
- **Grad-CAM** permet de verifier que le modele regarde les bonnes regions
- Le pipeline est **reproductible** et bien documente

---

<div align="center">

**Projet B3 Deep Learning** - 2026

Made with ❤️ and PyTorch

</div>
