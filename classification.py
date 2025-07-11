import pickle

# Muat model dan TF-IDF vectorizer
svm_model = pickle.load(open('svm_sentiment_model.pkl', 'rb'))
svm_vectorizer = pickle.load(open('tfidf_vectorizer_svm.pkl', 'rb'))

rf_model = pickle.load(open('random_forest_sentiment_model.pkl', 'rb'))
rf_vectorizer = pickle.load(open('tfidf_vectorizer_rf.pkl', 'rb'))

# Mapping label angka ke label string
label_map = {
    -1: "negative",
     0: "neutral",
     1: "positive"
}

def classify_texts(texts, model_name):
    if model_name == 'svm':
        X = svm_vectorizer.transform(texts)
        preds = svm_model.predict(X)
    elif model_name == 'rf':
        X = rf_vectorizer.transform(texts)
        preds = rf_model.predict(X)
    else:
        raise ValueError("Model tidak dikenali")

    # Konversi hasil prediksi ke label string
    return [label_map[int(p)] for p in preds]
