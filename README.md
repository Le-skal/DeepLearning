<div align="center">

# 🫁 Pneumonia Detection CNN
### *Classification d'Images Medicales par Deep Learning*

<p><em>Detecter la pneumonie sur des radiographies thoraciques avec des reseaux de neurones convolutifs</em></p>

![Status](https://img.shields.io/badge/status-operational-success?style=flat)
![Accuracy](https://img.shields.io/badge/accuracy-91.03%25-blue?style=flat)
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
Flatten -> Dense(128) + Dropout -> Dense(1)
```

### ResNet18 (Transfer Learning)
- ResNet18 pre-entraine sur ImageNet
- Couches convolutives gelees
- Nouvelle tete de classification fine-tunee

## 📈 Resultats

### Comparaison des Modeles

| Metrique | CNN Baseline | ResNet18 | Amelioration |
|----------|--------------|----------|--------------|
| **Accuracy** | 76.44% | **91.03%** | +14.59% |
| **Precision** | 72.97% | **91.96%** | +18.99% |
| **Recall** | 98.97% | 93.85% | -5.12% |
| **Specificite** | 38.89% | **86.32%** | +47.43% |
| **F1-Score** | 84.00% | **92.89%** | +8.89% |

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

## 🔬 Discussion Critique des Limites

### Limites du Dataset

| Probleme | Impact | Piste d'amelioration |
|----------|--------|----------------------|
| **Set de validation trop petit** (16 images) | Difficulte a detecter l'overfitting pendant l'entrainement, metriques de validation instables | Fusionner val+test et faire un split 80/20, ou utiliser la cross-validation |
| **Desequilibre des classes** (3:1 pneumonia/normal) | Le modele peut etre biaise vers la classe majoritaire | Utiliser class weights, oversampling (SMOTE), ou undersampling |
| **Source unique** | Les images proviennent d'un seul hopital (Guangzhou) - biais demographique et materiel | Valider sur des datasets externes (NIH ChestX-ray14, CheXpert) |

### Limites du Modele

- **Pas de distinction bacterien/viral** : Le dataset contient des pneumonies bacteriennes et virales melangees, mais le modele ne les distingue pas
- **Sensibilite au preprocessing** : Les performances dependent fortement du redimensionnement et de la normalisation
- **Boite noire** : Meme avec Grad-CAM, on ne sait pas exactement quels patterns radiologiques le modele detecte

### Limites Methodologiques

- **Pas de validation croisee** : Les resultats peuvent varier selon le split train/test
- **Hyperparametres non optimises** : Pas de grid search ou random search systematique
- **Seuil de decision fixe a 0.5** : En contexte medical, un seuil plus bas pourrait etre preferable pour maximiser le recall (eviter les faux negatifs)

### Limites pour un Usage Clinique

- **Pas de calibration des probabilites** : Les scores de confiance ne refletent pas les vraies probabilites
- **Pas de gestion de l'incertitude** : Le modele donne toujours une prediction, meme sur des images hors distribution
- **Absence de validation clinique** : Aucun test sur des cas reels avec verification par des radiologues

> **Note** : Ces limites sont normales pour un projet pedagogique de 10 jours. Un deploiement clinique reel necessiterait des validations beaucoup plus poussees.

## 🎓 Conclusion

- Le **Transfer Learning avec ResNet18** ameliore significativement les performances (+15% accuracy)
- La **specificite** passe de 44% a 86% (meilleure detection des cas normaux)
- **Grad-CAM** permet de verifier que le modele regarde les bonnes regions
- Le pipeline est **reproductible** et bien documente

---

<div align="center">

**Projet B3 Deep Learning** - 2026

Made with ❤️ and PyTorch

</div>
