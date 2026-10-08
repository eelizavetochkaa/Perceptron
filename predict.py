import csv
import argparse
import numpy as np
from mlp import MLP

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='data/validation.csv')
parser.add_argument('--model', default='model/model.npz')
parser.add_argument('--scaler', default='model/scaler.npz')
parser.add_argument('--output', default='predictions.csv')
args = parser.parse_args()

DATA_FILE = args.dataset
MODEL_FILE = args.model
SCALER_FILE = args.scaler
OUTPUT_FILE = args.output


def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    try:
        float(rows[0][2])
    except ValueError:
        rows = rows[1:]
    ids = [row[0] for row in rows]
    true_labels = [row[1] for row in rows]
    X = np.array([[float(v) for v in row[2:]] for row in rows], dtype=float)
    return ids, true_labels, X


def binary_cross_entropy(y_true, probabilities):
    eps = 1e-12 
    y = np.array([0 if label == 'B' else 1 for label in y_true])
    p = probabilities[:, 1]
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

ids, true_labels, X = load_csv(DATA_FILE)

scaler = np.load(SCALER_FILE)
X = (X - scaler['mean']) / scaler['std']

model = MLP.load(MODEL_FILE)
probabilities = model.predict_proba(X)
predicted_indices = np.argmax(probabilities, axis=1)
predicted_labels = ['B' if i == 0 else 'M' for i in predicted_indices]

true_indices = np.array([0 if label == 'B' else 1 for label in true_labels])
accuracy = np.mean(true_indices == predicted_indices)
bce = binary_cross_entropy(true_labels, probabilities)

with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'true_diagnosis', 'predicted_diagnosis', 'P(B)', 'P(M)'])
    for i in range(len(ids)):
        writer.writerow([
            ids[i], true_labels[i], predicted_labels[i],
            f'{probabilities[i, 0]:.6f}', f'{probabilities[i, 1]:.6f}'
        ])

print('Результаты работы программы прогнозирования ')
print(f'Обработано образцов: {len(X)}')
print(f'Точность (Accuracy): {accuracy:.4f}')
print(f'Бинарная кросс-энтропия (BCE Loss): {bce:.4f}')
print(f'Прогнозы успешно сохранены в файл: {OUTPUT_FILE}')