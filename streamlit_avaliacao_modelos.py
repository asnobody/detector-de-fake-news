import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Exibir métricas de avaliação dos modelos
if st.checkbox("Mostrar avaliação dos modelos"):
    st.subheader("📊 Avaliação dos Modelos")

    df = load_data()
    df["text"] = df["text"].apply(preprocess_text)
    X = df["text"]
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=0),
        "Random Forest": RandomForestClassifier(random_state=0)
    }

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.write(f"**{name}** - Acurácia: {acc:.2f}")

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=["Falsa", "Verdadeira"],
                    yticklabels=["Falsa", "Verdadeira"])
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de Confusão - {name}")
        st.pyplot(fig)