import os
import csv
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = 'data/data.csv'

def load_data(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    labels = np.array([r[1] for r in rows])
    X = np.array([[float(v) for v in r[2:]] for r in rows])
    return labels, X

os.makedirs('plots', exist_ok=True)

labels, X = load_data(DATA_FILE)

print('Количество объектов:', len(X))
print('Количество признаков:', X.shape[1])
print('Количество пропущенных значений:', np.isnan(X).sum())
print('Доброкачественные случаи (B):', np.sum(labels == 'B'))
print('Злокачественные случаи (M):', np.sum(labels == 'M'))

print('\nСредние значения признаков (первые 5):')
print(X.mean(axis=0)[:5])

print('\nСтандартные отклонения признаков (первые 5):')
print(X.std(axis=0)[:5])

plt.figure()
unique, counts = np.unique(labels, return_counts=True)
plt.bar(unique, counts)
plt.xlabel('Диагноз')
plt.ylabel('Количество объектов')
plt.title('Распределение классов')
plt.tight_layout()
plt.savefig('plots/class_distribution.png', dpi=150)
plt.close()

corr = np.corrcoef(X, rowvar=False)

plt.figure(figsize=(8, 7))
plt.imshow(corr, aspect='auto')
plt.colorbar(label='Коэффициент корреляции')
plt.xlabel('Номер признака')
plt.ylabel('Номер признака')
plt.title('Матрица корреляции признаков')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png', dpi=150)
plt.close()

print('\nАнализ данных завершён.')
print('Графики сохранены в папке plots/.')