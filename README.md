# EEG-based Mental State Classification 🧠

This project uses EEG signals to classify mental states (e.g., focused, relaxed, stressed) using machine learning.


## 📂 Dataset

The dataset is automatically fetched from the UCI Machine Learning Repository using the `ucimlrepo` package:

```python
from ucimlrepo import fetch_ucirepo
eeg_eye_state = fetch_ucirepo(id=264)
X = eeg_eye_state.data.features
y = eeg_eye_state.data.targets

## 🔧 Tech Stack
- Python, Pandas, Scikit-learn, Matplotlib
- Preprocessing: Bandpass filtering, normalization
- ML models: SVM, Random Forest

## 📊 Results
Achieved ~85% accuracy in detecting eye states.

## 📁 Structure
- `notebooks/`: Development and analysis
- `src/`: Data processing and model training scripts
- `data/`: Raw datasets (ignored in repo)

## 🚀 Run
```bash
pip install -r requirements.txt
jupyter notebook

