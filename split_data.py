import csv
import random
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.8
DATA_DIR = Path('data')
FEATURE_NAMES = [f'метрика_{i}' for i in range(1, 31)]

def read_data(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if not row:
                continue
            rows.append([row[0], row[1]] + row[2:])
    return rows

def main():
    rows = read_data(DATA_DIR / 'data.csv')
    rng = random.Random(SEED)
    
   
    benign = [r for r in rows if r[1] == 'B']
    malignant = [r for r in rows if r[1] == 'M']
    rng.shuffle(benign)
    rng.shuffle(malignant)

    def split_group(group):
        n_train = int(len(group) * TRAIN_RATIO)
        return group[:n_train], group[n_train:]

    benign_train, benign_val = split_group(benign)
    malignant_train, malignant_val = split_group(malignant)

    train = benign_train + malignant_train
    validation = benign_val + malignant_val
    rng.shuffle(train)
    rng.shuffle(validation)

    header = ['id', 'diagnosis'] + FEATURE_NAMES
    
    for filename, data in [('train.csv', train), ('validation.csv', validation)]:
        with open(DATA_DIR / filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

    print(f'Всего записей обработано: {len(rows)}')
    print(f'Размер обучающей выборки (Train): {len(train)}')
    print(f'Размер валидационной выборки (Validation): {len(validation)}')
    print(f'Значение Seed для рандома: {SEED}')
    print('Успешно! Файлы сохранены: data/train.csv и data/validation.csv')

if __name__ == '__main__':
    main()