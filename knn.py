import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Criar diretório para imagens caso não exista
if not os.path.exists("imagens"):
    os.makedirs("imagens")

# Carregar o conjunto de dados
data = pd.read_csv('breastcancer.csv')

# Separar atributos (X) e rótulos (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Funções para normalização
def normalize_train(X_train):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    return (X_train - mean) / std, mean, std

def normalize_test(X_test, mean, std):
    return (X_test - mean) / std

# Implementação do KNN
class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.array([self.euclidean_distance(x, x_train) for x_train in self.X_train])
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            unique, counts = np.unique(k_labels, return_counts=True)
            predictions.append(unique[np.argmax(counts)])
        return np.array(predictions)

# Validação cruzada com 10 folds
def cross_validate(model, X, y):
    n_samples = len(X)
    fold_size = n_samples // 10
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    accuracies, precisions, recalls, f1_scores = [], [], [], []

    for i in range(10):
        test_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, test_indices)

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        X_train, mean, std = normalize_train(X_train)
        X_test = normalize_test(X_test, mean, std)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        tn = np.sum((y_pred == 0) & (y_test == 0))

        accuracy = (tp + tn) / len(y_test)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": (np.mean(accuracies), np.std(accuracies)),
        "precision": (np.mean(precisions), np.std(precisions)),
        "recall": (np.mean(recalls), np.std(recalls)),
        "f1_score": (np.mean(f1_scores), np.std(f1_scores))
    }

# Executar KNN
knn = KNN(k=3)
results = cross_validate(knn, X, y)

# Exibir resultados
print("Resultados para KNN (k=3):")
for metric, values in results.items():
    print(f"  {metric.capitalize()}: Média = {values[0]:.4f}, Desvio Padrão = {values[1]:.4f}")


# Criar um único gráfico com todas as métricas
plt.figure(figsize=(8, 6))

metrics = ["accuracy", "precision", "recall", "f1_score"]
means = [results[metric][0] for metric in metrics]
stds = [results[metric][1] for metric in metrics]

# Criar barras com erro padrão
plt.bar(metrics, means, yerr=stds, capsize=5, color=['blue', 'green', 'red', 'purple'])

# Adicionar título e rótulos
plt.title("Desempenho do KNN (k=3)")
plt.ylabel("Média das métricas")
plt.ylim(0, 1)  # Como as métricas variam de 0 a 1, definir o limite melhora a visualização

# Salvar e fechar o gráfico
plt.tight_layout()
plt.savefig("imagens/knn_metrics.png")
plt.close()
