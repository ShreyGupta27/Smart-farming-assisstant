# 🌾 AgriAI – Intelligent Agricultural Decision Support System

A comprehensive machine learning project designed to assist farmers and agricultural stakeholders with data-driven decision-making across five core farming challenges: crop selection, yield prediction, disease detection, fertilizer recommendation, and irrigation management.

---

## Project Overview

AgriAI brings together five machine learning notebooks that together form an end-to-end intelligent agriculture system. Each module tackles a distinct problem in precision farming, using classical ML, ensemble methods, and deep learning where appropriate.

---

## Modules

### 1. 🌱 Crop Recommendation (`Crop_Recommendation.ipynb`)
Recommends the most suitable crop to grow based on soil and environmental parameters.

**Input features:** Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, rainfall  
**Model:** Random Forest Classifier  
**Approach:** EDA with pairplots and correlation heatmaps, followed by model training and evaluation with accuracy score, classification report, and confusion matrix.

---

### 2. 📈 Crop Yield Prediction (`Crop_Yield_Prediction.ipynb`)
Predicts the expected yield for a given crop under specific soil and climate conditions.

**Input features:** N, P, K, temperature, humidity, pH, rainfall  
**Models compared:** Logistic Regression, Random Forest Classifier, XGBoost  
**Approach:** Feature scaling with `StandardScaler`, label encoding, multi-model comparison with confusion matrices for each.

---

### 3. 🦠 Crop Disease Detection (`Crop_Disease_Detection.ipynb`)
Detects plant diseases from leaf images using a Convolutional Neural Network (CNN).

**Dataset:** PlantVillage Dataset (via Kaggle)  
**Model:** Custom CNN built with TensorFlow/Keras (Conv2D + MaxPooling layers)  
**Approach:** Image data augmentation using `ImageDataGenerator`, 80/20 train-validation split, trained on colour images resized to 224×224.

---

### 4. 🧪 Fertilizer Prediction (`Fertilizer_Prediction.ipynb`)
Recommends the most suitable fertilizer based on soil composition, crop type, and environmental conditions.

**Input features:** Temperature, humidity, moisture, soil type, crop type, N, P, K  
**Models compared:** 20+ classifiers including Logistic Regression, Random Forest, Decision Tree, SVM, KNN, MLP, LDA, QDA, CatBoost, LightGBM, XGBoost, GradientBoosting, AdaBoost, and more  
**Approach:** Comprehensive EDA, encoding of categorical variables (soil type, crop type), and a thorough multi-model benchmark.

---

### 5. 💧 Intelligent Irrigation System (`Intelligent_Irrigation_System.ipynb`)
Predicts whether the irrigation pump should be switched ON or OFF based on real-time sensor data.

**Input features:** Soil moisture, temperature, humidity, and other sensor readings  
**Target:** Pump status (binary: 1 = ON, 0 = OFF)  
**Models compared:** Random Forest, SVM, Decision Tree, KNN, Logistic Regression  
**Approach:** EDA with scatter plots, feature scaling, and evaluation with accuracy, precision, recall, and F1-score.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn`, `xgboost`, `lightgbm`, `catboost` |
| Deep Learning | `TensorFlow`, `Keras` |
| Image Processing | `Pillow (PIL)` |
| Model Persistence | `joblib`, `pickle` |
| Environment | Google Colab / Kaggle Notebooks |

---

## Dataset Sources

| Module | Dataset |
|---|---|
| Crop Recommendation | [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) |
| Crop Yield Prediction | Same as Crop Recommendation (`Crop_recommendation.csv`) |
| Crop Disease Detection | [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) |
| Fertilizer Prediction | [Fertilizer Prediction Dataset](https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction) |
| Intelligent Irrigation | Soil sensor dataset (`data.csv`) |

---

## Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost tensorflow pillow joblib ipywidgets
```

### Running on Google Colab / Kaggle

Each notebook is self-contained and designed to run on Google Colab or Kaggle. Simply:

1. Upload the notebook to Colab or open it on Kaggle.
2. Place the required dataset (`.zip` or `.csv`) in the same directory or mount your Google Drive.
3. For `Crop_Disease_Detection.ipynb`, configure your `kaggle.json` API key to download the PlantVillage dataset automatically.
4. Run cells top to bottom.

> **Note:** For the disease detection notebook, ensure you have a GPU available or reduce the model complexity for CPU-only environments.

---

## Results Summary

| Module | Best Model | Metric |
|---|---|---|
| Crop Recommendation | Random Forest | High accuracy (~99%) |
| Crop Yield Prediction | XGBoost / Random Forest | Competitive across all three models |
| Crop Disease Detection | Custom CNN | Validated on PlantVillage (38 classes) |
| Fertilizer Prediction | CatBoost / LightGBM | Benchmarked across 20+ classifiers |
| Intelligent Irrigation | Random Forest / SVM | Evaluated on Accuracy, Precision, Recall, F1 |

---

## License

Copyright (c) 2026 Shrey Gupta. All rights reserved.

Unauthorized copying, modification, or distribution of this software, via any medium, is strictly prohibited.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Acknowledgements

- [Kaggle](https://www.kaggle.com) for hosting the datasets
- Scikit-learn, TensorFlow, and the open-source ML community
