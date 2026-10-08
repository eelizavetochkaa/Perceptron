import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP

parser = argparse.ArgumentParser()
parser.add_argument('--train', default='data/train.csv')
parser.add_argument('--valid', default='data/validation.csv')
parser.add_argument('--model', default='model/model.npz')
parser.add_argument('--scaler', default='model/scaler.npz')
parser.add_argument('--layer', type=int, nargs='+', default=[32, 16])
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

TRAIN_FILE = args.train
VAL_FILE = args.valid
MODEL_FILE = args.model
SCALER_FILE = args.scaler

LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
SEED = args.seed


def load_csv(path):
    """Загрузка данных из CSV файла и бинаризация целевого признака"""
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    try:
        float(rows[0][2])
    except ValueError:
        rows = rows[1:]
    X = np.array([[float(v) for v in row[2:]] for row in rows], dtype=float)
    y = np.array([0 if row[1] == 'B' else 1 for row in rows], dtype=int)
    return X, y


def one_hot(y):
    """Преобразование меток классов в формат One-Hot encoding."""
    result = np.zeros((len(y), 2))
    result[np.arange(len(y)), y] = 1
    return result


def standardize_train(X):
    """Z-score нормализация признаков """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    return (X - mean) / std, mean, std


def accuracy(y_true, y_pred):
    """Вычисление метрики точности (Accuracy)."""
    return np.mean(y_true == y_pred)

os.makedirs(os.path.dirname(MODEL_FILE) or '.', exist_ok=True)
os.makedirs(os.path.dirname(SCALER_FILE) or '.', exist_ok=True)
os.makedirs('plots', exist_ok=True)

X_train, y_train = load_csv(TRAIN_FILE)
X_val, y_val = load_csv(VAL_FILE)
print(f'x_train shape : {X_train.shape}')
print(f'x_valid shape : {X_val.shape}')
X_train, mean, std = standardize_train(X_train)
X_val = (X_val - mean) / std
np.savez(SCALER_FILE, mean=mean, std=std)

y_train_oh = one_hot(y_train)
y_val_oh = one_hot(y_val)

model = MLP(input_size=X_train.shape[1], hidden_sizes=args.layer, seed=SEED)
rng = np.random.default_rng(SEED)

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(1, EPOCHS + 1):
    indices = rng.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train_oh[indices]

    for start in range(0, len(X_train), BATCH_SIZE):
        end = start + BATCH_SIZE
        model.train_step(X_train_shuffled[start:end], y_train_shuffled[start:end], LEARNING_RATE)

    train_prob = model.predict_proba(X_train)
    val_prob = model.predict_proba(X_val)

    train_loss = model.cross_entropy(y_train_oh, train_prob)
    val_loss = model.cross_entropy(y_val_oh, val_prob)
    train_acc = accuracy(y_train, np.argmax(train_prob, axis=1))
    val_acc = accuracy(y_val, np.argmax(val_prob, axis=1))

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f'Эпоха {epoch:03d}/{EPOCHS} | '
          f'Обучение: Loss={train_loss:.4f}, Acc={train_acc:.4f} | '
          f'Валидация: Loss={val_loss:.4f}, Acc={val_acc:.4f}')

model.save(MODEL_FILE)
# График 1: Функция потерь
plt.figure()
plt.plot(train_losses, label='Loss на обучении')
plt.plot(val_losses, label='Loss на валидации')
plt.xlabel('Эпоха')
plt.ylabel('Кросс-энтропия')
plt.title('Кривая обучения: Функция потерь')
plt.legend()
plt.tight_layout()
plt.savefig('plots/loss.png', dpi=150)
plt.close()

# График 2: Точность
plt.figure()
plt.plot(train_accs, label='Точность на обучении')
plt.plot(val_accs, label='Точность на валидации')
plt.xlabel('Эпоха')
plt.ylabel('Точность (Accuracy)')
plt.title('Кривая обучения: Точность')
plt.legend()
plt.tight_layout()
plt.savefig('plots/accuracy.png', dpi=150)
plt.close()
print('\nОбучение успешно завершено')
print('Модель сохранена в файл:', MODEL_FILE)
print(f'Итоговая точность на валидации: {val_accs[-1]:.4f}')
print(f'Итоговая кросс-энтропия на валидации: {val_losses[-1]:.4f}')