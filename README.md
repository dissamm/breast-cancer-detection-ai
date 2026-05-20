# 🔬 Breast Cancer Detection AI - Research Prototype

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28.0-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## ⚠️ IMPORTANT DISCLAIMER

**This is a RESEARCH PROTOTYPE, NOT a medical device.**

- Not FDA-approved or clinically validated
- Not intended for clinical decision-making
- For research and educational purposes only

## 📊 Model Performance

| Threshold | Sensitivity | Specificity | False Negatives | False Positives |
|-----------|-------------|-------------|-----------------|-----------------|
| 0.5 (default) | 93.8% | 99.1% | 4 | 1 |
| 0.3 (balanced) | 96.9% | 97.2% | 2 | 3 |
| 0.2 (sensitive) | 98.4% | 91.6% | 1 | 10 |

**ROC-AUC:** 0.990 | **Test Accuracy:** 97.1%

## 🚀 Live Demo

[Click here to try the app](https://breast-cancer-detection-ai-YOUR_USERNAME.streamlit.app)

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML Model:** Random Forest
- **Visualization:** Plotly
- **Data:** Wisconsin Breast Cancer Dataset (UCI)

## 📁 Project Structure
