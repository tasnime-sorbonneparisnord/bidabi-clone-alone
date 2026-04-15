# bidabi-clone-adapt-create
# BIDABI : Clone → Adapt → Create

Dépôt pédagogique du cours **Big Data and Business Intelligence (BIDABI)**.  
Ce projet a pour objectif d’initier les étudiants au travail avec du code open‑source, à l’adaptation de projets existants et à la création de leur propre jeu de données d’images.

## 🎯 Objectif du dépôt
Ce dépôt sert de **plateforme d’apprentissage** où les étudiants réalisent un cycle complet de travail en data et en machine learning :

- cloner un projet open‑source depuis GitHub
- analyser sa structure, ses dépendances et son fonctionnement
- adapter le code à un nouveau contexte
- créer un jeu de données d’images personnalisé
- intégrer ce jeu de données dans un pipeline ML existant

L’objectif est de reproduire des situations réelles rencontrées par les ingénieurs data et ML lorsqu’ils doivent réutiliser et modifier du code provenant d’autres développeurs.

## 🎓 Public visé
Ce projet est destiné aux étudiants du cours **BIDABI**, notamment ceux qui s’intéressent à :

- l’apprentissage automatique
- l’ingénierie des données
- la reproductibilité des expériences
- l’utilisation de GitHub et des projets open‑source

## 🧩 Contenu du dépôt
Le dépôt inclura :

- des exemples de code à analyser et adapter
- un modèle de structure pour le jeu de données
- des consignes pour les travaux pratiques
- des instructions pour exécuter et modifier le projet

## 🛠️ Compétences développées
Les étudiants apprendront à :

- lire et comprendre du code écrit par d’autres
- manipuler des dépôts GitHub
- concevoir et organiser un jeu de données d’images
- intégrer des données dans un pipeline ML
- documenter leur travail de manière claire et reproductible

## 📄 Licence et usage
Ce dépôt est destiné **exclusivement à des fins pédagogiques** dans le cadre du cours BIDABI.  
Le code et les ressources peuvent être simplifiés ou modifiés pour faciliter l’apprentissage.

## 🚀 Version 3.0 — Pipeline complet et reproductible
Cette version 3.0 ajoute un pipeline d’entraînement complet pour un modèle de classification d’images basé sur ResNet-18.

### Contenu de la version 3.0
- dataset RAW versionné avec DVC via `data/raw.dvc`
- code d’entraînement finalisé dans `src/classificator.py`
- préparation des splits `train/val/test`
- modèle entraîné enregistré sous `model/best_model_resnet18_finetuned.pth`
- pipeline DVC défini dans `dvc.yaml`
- instructions de reproduction dans ce README

### Reproduction
1. Créez un environnement virtuel propre :
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Installez les dépendances :
   ```powershell
   pip install -r requirements.txt
   ```
3. Récupérez les données versionnées avec DVC :
   ```powershell
   dvc pull
   ```
4. Si le dossier `data/raw` existe, préparez les splits :
   ```powershell
   python src/data_prepare.py --raw-dir data/raw --output-dir data --val-ratio 0.2 --test-ratio 0.2 --seed 42
   ```
5. Lancez l’entraînement :
   ```powershell
   python main.py
   ```

### Résultats
- modèle : `model/best_model_resnet18_finetuned.pth`
- métriques : `model/metrics.json`
- graphiques de progression : `model/plots/`

### Versioning Git
- tag Git attendu : `v3.0`
- release GitHub attendue : **Version 3.0**
  - description du pipeline d’entraînement
  - version du dataset utilisée
  - modèle entraîné versionné avec DVC
  - instructions de reproduction

