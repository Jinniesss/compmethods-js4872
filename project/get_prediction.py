import json
import requests

HOST = 'http://127.0.0.1:8888'
K = 5
PCA_N = 30
OUTPUT_PATH = 'project/results_predict.json'

url = f"{HOST.rstrip('/')}/api/predict?k={K}&pca_n={PCA_N}"
r = requests.get(url, timeout=60)
r.raise_for_status()
data = r.json()

with open(OUTPUT_PATH, 'w') as f:
    json.dump(data, f)

print(f"Saved prediction JSON to {OUTPUT_PATH}")
