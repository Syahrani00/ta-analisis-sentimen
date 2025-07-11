from flask import Blueprint, jsonify
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from main import db
from models import DataHasil

visual_bp = Blueprint("visual", __name__)

@visual_bp.route("/visualisasi", methods=["GET"])
def visualisasi():
    hasil = DataHasil.query.all()
    if not hasil:
        return jsonify({"error": "Belum ada data hasil klasifikasi"}), 400

    tweets = [row.tweet for row in hasil]
    sentiments = [row.sentiment for row in hasil]

    df = pd.DataFrame({"tweet": tweets, "sentiment": sentiments})

    text_all = " ".join(df["tweet"])
    wc_all = WordCloud(width=800, height=400, background_color="white").generate(text_all)
    wc_all.to_file("static/wordcloud_all.png")

    for label in ["positive", "negative"]:
        text = " ".join(df[df.sentiment == label]["tweet"])
        if text:
            wc = WordCloud(width=800, height=400, background_color="white").generate(text)
            wc.to_file(f"static/wordcloud_{label}.png")

    sentiment_counts = df["sentiment"].value_counts()
    plt.figure(figsize=(6, 6))
    sentiment_counts.plot.pie(autopct="%1.1f%%", startangle=90, colors=["green", "red", "grey"])
    plt.title("Distribusi Sentimen")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("static/pie_chart.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="sentiment", palette="Set2")
    plt.title("Jumlah Tweet per Sentimen")
    plt.tight_layout()
    plt.savefig("static/bar_chart.png")
    plt.close()

    return jsonify({
        "message": "Visualisasi berhasil dibuat",
        "images": [
            "/static/wordcloud_all.png",
            "/static/wordcloud_positive.png",
            "/static/wordcloud_negative.png",
            "/static/pie_chart.png",
            "/static/bar_chart.png"
        ]
    }), 200