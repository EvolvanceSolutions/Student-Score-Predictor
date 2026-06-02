# Student-Score-Predictor
A custom neural network from scratch using NumPy Forward propagation Backpropagation Gradient descent Data normalization Train/test split without leakage Evaluation metrics (RMSE, MAE, R²) Real-world prediction pipeline  This is not just “vibe coding.” The architecture and logic are correct.
# 🎓 Student Score Predictor — Neural Network from Scratch

 A feedforward neural network built with pure NumPy (no TensorFlow, no PyTorch) that predicts a student's exam score and flags at-risk students before exam day.

---

## 📌 Overview

This project builds a regression neural network entirely from scratch — forward pass, backpropagation, and gradient descent — without any ML framework. Given four student habit inputs, the model predicts a final exam score with **R² = 0.9256** (92.5% accuracy) and an average error of only **±3.5 marks**.

The end result is a working **early warning system**: the model classifies each student as 🟢 ON TRACK, 🟡 MONITOR, or 🔴 AT RISK so teachers can intervene before exam day.

---

## 🧠 What I Built

| Component | Detail |
|---|---|
| Architecture | 4 → 8 → 1 (49 total parameters) |
| Activation | ReLU (hidden layer) · Linear (output) |
| Loss function | Mean Squared Error (MSE) |
| Optimizer | Vanilla Gradient Descent |
| Learning rate | 0.01 |
| Epochs | 5,000 |

---

## 📊 Results

| Metric | Value |
|---|---|
| R² Score | **0.9256** |
| RMSE | **3.54 marks** |
| MAE | **3.47 marks** |
| Loss reduction | 0.742 → 0.005 **(99.3% drop)** |

---

## 🗂️ Input Features

The model takes four inputs per student:

| Feature | Range | Description |
|---|---|---|
| `study_hours` | 1 – 10 hrs | Daily study time |
| `sleep_hours` | 4 – 9 hrs | Nightly sleep |
| `prev_score` | 40 – 95 | Previous exam result |
| `attendance` | 50 – 100 % | Class attendance rate |

**Target:** `final_score` (0 – 100 marks)

---

## 🔑 Key Concepts Demonstrated

- **Manual backpropagation** — chain rule implemented step by step
- **Data leakage prevention** — split before normalisation; test stats computed from training data only
- **Min-max normalisation** — applied correctly across train/test splits
- **Weight initialisation** — He initialisation for ReLU layers
- **Model persistence** — weights saved as `.npy` files for reuse without retraining
- **Prediction pipeline** — denormalisation and risk classification in a single function

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/student-score-predictor.git
cd student-score-predictor
```

### 2. Install dependencies
```bash
pip install numpy matplotlib
```

### 3. Open the notebook
```bash
jupyter notebook project_01_student_score_predictor.ipynb
```

### 4. Run all cells in order (Cell 1 → Cell 8)

---

## 🔮 Try a Prediction

The last cell lets you predict any student's score:

```python
predict_and_explain(
    study_h    = 4,
    sleep_h    = 6,
    prev_sc    = 58,
    attend_pct = 70
)
```

**Output:**
```
==================================================
  PREDICTED SCORE : 60.4/100  [MONITOR]
==================================================
  Strongest factor : Attendance
  Weakest factor   : Study hours
  Model confidence : R² = 0.9256 (92.5% accurate)

  RECOMMENDATION : Weekly check-in advised.
```

---

## ♻️ Reload Saved Weights

The trained weights are saved as `.npy` files. To reload without retraining:

```python
model    = NeuralNetwork()
model.W1 = np.load('p01_W1.npy')
model.b1 = np.load('p01_b1.npy')
model.W2 = np.load('p01_W2.npy')
model.b2 = np.load('p01_b2.npy')
X_min    = np.load('p01_X_min.npy')
X_max    = np.load('p01_X_max.npy')
y_min    = float(np.load('p01_y_min.npy'))
y_max    = float(np.load('p01_y_max.npy'))
```

---

## 📁 Project Structure

```
student-score-predictor/
├── project_01_student_score_predictor.ipynb   ← main notebook
├── p01_W1.npy                                 ← trained weights
├── p01_b1.npy
├── p01_W2.npy
├── p01_b2.npy
├── p01_X_min.npy                              ← normalisation stats
├── p01_X_max.npy
├── p01_y_min.npy
├── p01_y_max.npy
└── README.md
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-only-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-visualisation-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)

- NumPy — all matrix operations, forward pass, backprop
- Matplotlib — loss curves, prediction scatter plots
- No scikit-learn, no PyTorch, no TensorFlow

---

## 💡 Why This Project Matters

Most people learn neural networks by calling `model.fit()`. This project proves the understanding goes deeper — every gradient, every weight update, every normalisation step is written by hand. The 49 parameters in this network genuinely learned a relationship from 800 examples with zero guidance from an ML library.

---

## 👤 Author

Your Name:- Anekant
[LinkedIn](https://www.linkedin.com/company/evolvance-edtech24/) · [GitHub](https://github.com/evolvancesolutions) · [Portfolio](https://evolvance.solutions)

---

*Part of a series of ML projects built from scratch — Project 01 of 10.*
