import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Paso 1: Recolección de datos
# Supongamos que ya tienes un DataFrame llamado "data" con una columna "sentence" que contiene las oraciones.

# Paso 2: Preprocesamiento de datos
nltk.download('punkt')
data['sentence'] = data['sentence'].str.lower()
data['sentence'] = data['sentence'].apply(nltk.word_tokenize)

# Paso 3: Entrenamiento del modelo
# Primero, necesitas un conjunto de datos etiquetado que contenga oraciones con bullying y sin bullying.
# Divide tus datos en conjuntos de entrenamiento y prueba.
X = data['sentence']
y = data['bullying_label']  # Columna que contiene las etiquetas de bullying (1 para bullying, 0 para no bullying)

# Utiliza un vectorizador TF-IDF para convertir el texto en características numéricas.
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Entrena un modelo de clasificación, por ejemplo, SVM (Support Vector Machine).
model = SVC(kernel='linear')
model.fit(X_tfidf, y)

# Paso 4: Clasificación de oraciones
# Supongamos que tienes nuevas oraciones en una lista llamada "new_sentences".
new_sentences = ["Esta es una oración sin bullying.", "Eres tan tonto, deberías irte.", "Deberías dejar de hacer eso."]
new_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in new_sentences]
new_sentences_tfidf = tfidf_vectorizer.transform(new_sentences)
predictions = model.predict(new_sentences_tfidf)

# Paso 5: Generar el reporte en un archivo Excel
# Crea un DataFrame con las oraciones y sus etiquetas de bullying.
report_data = {'Sentence': new_sentences, 'Bullying_Label': predictions}
report_df = pd.DataFrame(report_data)

# Guarda el DataFrame en un archivo Excel.
report_df.to_excel('reporte_bullying.xlsx', index=False)
