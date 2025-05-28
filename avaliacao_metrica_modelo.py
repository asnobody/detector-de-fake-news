from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Treinando o modelo (exemplo com Random Forest)
from sklearn.ensemble import RandomForestClassifier

# Pré-processamento e vetorização
df["text"] = df["text"].apply(preprocess_text)
X = df["text"]
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Treinando o modelo
model = RandomForestClassifier(random_state=0)
model.fit(X_train_vec, y_train)

# Fazendo previsões
y_pred = model.predict(X_test_vec)

# Acurácia
acc = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acc:.2f}")

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)

# Visualização
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Falsa", "Verdadeira"], yticklabels=["Falsa", "Verdadeira"])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Random Forest')
plt.show()