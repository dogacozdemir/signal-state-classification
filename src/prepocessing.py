from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

def load_data():
    # dataset was selected based on structure compatability, not domain
    eeg_eye_state = fetch_ucirepo(id=264)
    X = eeg_eye_state.data.features
    y = eeg_eye_state.data.targets.values.ravel()  # label format adjusted
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
