# library
import os, glob, json, warnings, hashlib, shutil, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import random, collections
import librosa
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

# file datapath
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR        = PROJECT_DIR / "artist20"
TRAINVAL_DIR    = ROOT_DIR / "train_val"
TEST_DIR        = ROOT_DIR / "test"
VOCALS_OUT_ROOT = PROJECT_DIR / "separated"
MEL_DIR         = PROJECT_DIR / "mel_spec"   
# 輸出可放在專案資料夾，WSL 重開也不會消失
Path(VOCALS_OUT_ROOT).mkdir(parents=True, exist_ok=True)
Path(MEL_DIR).mkdir(parents=True, exist_ok=True)

# parameters
SR = 22050
DURATION = 30       
N_MFCC = 30
N_FFT = 4096        
HOP_LENGTH = 512
offset = 15    
  


def extract_features(filepath, sr=SR, offset=offset, duration=DURATION , pad_to_full=True):
    y, sr = librosa.load(filepath, sr=sr, mono=True, offset=offset, duration=duration)
    # （可選）若歌曲不夠長，補零到剛好 30 秒，避免不同長度的微差
    if pad_to_full and duration > 0:
        target = int(sr * duration)
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH) # N_MFCC=20 會自己先做STFT
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)# 12 會自己先做STFT
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spec_bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spec_con  = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85,
                                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    def agg_stats(feat):
        return np.hstack([np.mean(feat, axis=1), np.std(feat, axis=1), np.median(feat, axis=1)])

    
    return np.hstack([
        agg_stats(mfcc), agg_stats(chroma), agg_stats(spec_cent),
        agg_stats(spec_bw), agg_stats(spec_con), agg_stats(rolloff), agg_stats(zcr), agg_stats(rms)
    ])


# 將json檔的路徑分解後回傳
def load_split(json_path, root_dir=ROOT_DIR):
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        rel_paths = json.load(f)

    abs_paths = []
    for rp in rel_paths:
        # 去掉前導 "./" 後，與 root_dir 拼起來
        p = (Path(root_dir) / rp.lstrip("./")).resolve()
        abs_paths.append(str(p))
    return abs_paths

# 取得每首歌的singer
def get_label_from_path(path_str):
    p = Path(path_str)
    return p.parts[-3]

def build_dataset(path_list):
    X, y = [], []
    for p in path_list:
        try:
            feat = extract_features(str(p))   # librosa 接受字串路徑
            label = get_label_from_path(p)
            X.append(feat)
            y.append(label)
        except Exception as e:
            print("skip:", p, e)
    return np.array(X), np.array(y)

# main
train_json = os.path.join(ROOT_DIR, "train.json")
val_json   = os.path.join(ROOT_DIR, "val.json")

train_list = load_split(train_json)
val_list   = load_split(val_json)


X_train, y_train = build_dataset(train_list)
X_val,   y_val   = build_dataset(val_list)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
clf = RandomForestClassifier(
    n_estimators=100,   # 樹的數量
    max_depth=None,     # 樹的最大深度 (None = 自動)
    random_state=42
)
clf.fit(X_train, y_train)

# test
y_pred = clf.predict(X_val)
top1_accuracy = accuracy_score(y_val, y_pred)
print(f"Top-1 Accuracy: {top1_accuracy:.4f}")

# top3 accuracy
y_prob = clf.predict_proba(X_val)
# 若 y 是字串，要先轉 label index，才能對應 predict_proba 的順序
le = LabelEncoder()
le.fit(y_train)
y_val_encoded = le.transform(y_val)

top_k = 3
correct = 0

for i, probs in enumerate(y_prob):
    # 取機率最高的3個類別 index
    top_k_idx = np.argsort(probs)[-top_k:]
    if y_val_encoded[i] in top_k_idx:
        correct += 1

top3_accuracy = correct / len(y_val)
print(f"Top-3 Accuracy: {top3_accuracy:.4f}")

# Confusion matrix
print("Confusion matrix:")
# 先算混淆矩陣
cm = confusion_matrix(y_val, y_pred, labels=clf.classes_)

# 畫彩色熱圖
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
fig, ax = plt.subplots(figsize=(10, 10))  # 調整大小，避免文字擠在一起
disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
plt.title("Confusion Matrix (Validation Set)")
plt.xticks(rotation=90)   # x 軸標籤旋轉，避免歌手名字重疊
plt.show()
