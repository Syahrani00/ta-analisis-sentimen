from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import os
import pandas as pd
from models import db, DatasetMentah, DatasetPreprocessing, DatasetKlasifikasi
from preprocessing import preprocess_text
from classification import classify_texts
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from flask import send_from_directory


app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
db.init_app(app)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/')
def home():
    return render_template('home.html', active_page='home')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    data = None
    uploaded = False

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            df = pd.read_csv(filepath)

            # Validasi kolom wajib 'tweet'
            if 'tweet' not in df.columns:
                flash("File CSV harus memiliki kolom 'tweet'.", 'danger')
                return redirect(request.url)

            # Validasi jumlah baris minimal 100
            if len(df) < 100:
                flash("Dataset harus berisi minimal 100 data.", 'danger')
                return redirect(request.url)

            DatasetMentah.query.delete()
            db.session.commit()

            for _, row in df.iterrows():
                record = DatasetMentah(
                    username=row.get('username'),
                    tweet=row.get('tweet'),
                    waktu=row.get('waktu'),
                    tweet_link=row.get('tweet_link')
                )
                db.session.add(record)
            db.session.commit()

            # Set session upload_done
            session['upload_done'] = True
            session['upload_file'] = filepath

            columns_needed = ['username', 'tweet', 'waktu', 'tweet_link']
            for col in columns_needed:
                if col not in df.columns:
                    df[col] = ''  # tambahkan kolom kosong jika tidak ada

            data = df[columns_needed].fillna('').to_dict(orient='records')
            uploaded = True
            flash('File berhasil diupload dan data tersimpan di database.', 'success')
        else:
            flash('File harus berformat CSV.', 'danger')

    return render_template('upload.html', data=data, uploaded=uploaded, active_page='upload')





@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    if not session.get('upload_done'):
        flash('Selesaikan tahap Upload Dataset terlebih dahulu.', 'warning')
        return redirect(url_for('upload'))

    preprocessed_data = None  # default

    if request.method == 'POST':
        raw_data = DatasetMentah.query.all()

        keyword_lingkungan = [
            "gotong royong", "daur ulang sampah", "sungai bersih", "pengelolaan sampah",
            "gerakan hijau lingkungan", "lingkungan bersih", "pengolahan limbah",
            "pengelolaan sampah plastik", "pencemaran air", "bank sampah", "solusi banjir",
            "CSR lingkungan", "reboisasi", "penanaman pohon", "pembersihan sungai",
            "polusi udara", "sampah plastik", "banjir bandang", "kebakaran hutan",
            "limbah industri", "tumpukan sampah", "buang sampah sembarangan",
            "sungai tercemar", "tempat pembuangan sampah"
        ]

        def contains_keyword(tweet):
            tweet_lower = tweet.lower() if tweet else ''
            return any(kw in tweet_lower for kw in keyword_lingkungan)

        filtered = [d for d in raw_data if contains_keyword(d.tweet)]
        processed_records = []
        for d in filtered:
            text_clean = preprocess_text(d.tweet)
            processed_records.append({
                'id': d.id,
                'username': d.username,
                'tweet': d.tweet,
                'text_clean': text_clean,
                'waktu': d.waktu,
                'tweet_link': d.tweet_link  # tambahkan ini
            })

        preprocessed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'preprocessed.csv')
        pd.DataFrame(processed_records).to_csv(preprocessed_path, index=False)
        session['preprocessing_done'] = True
        session['preprocessed_file'] = preprocessed_path

        DatasetPreprocessing.query.delete()
        db.session.commit()

        for record in processed_records:
            db.session.add(DatasetPreprocessing(
                id=record['id'],
                username=record['username'],
                tweet=record['tweet'],
                text_clean=record['text_clean'],
                waktu=record['waktu'],
                tweet_link=record['tweet_link']  # pastikan ini juga di model
            ))
        db.session.commit()

        flash('Preprocessing selesai. Data disimpan dan siap diklasifikasi.', 'success')
        return render_template('preprocessing.html', preprocessed_data=processed_records, active_page='preprocessing')

    return render_template('preprocessing.html', preprocessed_data=preprocessed_data, active_page='preprocessing')


@app.route('/model', methods=['GET', 'POST'])
def model():
    if not session.get('preprocessing_done'):
        flash('Selesaikan tahap Preprocessing terlebih dahulu.', 'warning')
        return redirect(url_for('preprocessing'))

    if request.method == 'POST':
        model_name = request.form['model']
        # Ambil data preprocessing dari DB
        data_pre = DatasetPreprocessing.query.all()
        records = []
        for d in data_pre:
            records.append({
                'id': d.id,
                'username': d.username,
                'tweet': d.tweet,
                'text_clean': d.text_clean,
                'waktu': d.waktu,
                'tweet_link': d.tweet_link
            })
        df = pd.DataFrame(records)

        # Klasifikasi
        df['sentiment'] = classify_texts(df['text_clean'].tolist(), model_name)

        # Hapus data lama klasifikasi
        DatasetKlasifikasi.query.delete()
        db.session.commit()

        # Simpan hasil klasifikasi ke DB
        for _, row in df.iterrows():
            db.session.add(DatasetKlasifikasi(
                id=row['id'],
                username=row['username'],
                tweet=row['tweet'],
                text_clean=row['text_clean'],
                waktu=row['waktu'],
                sentiment=row['sentiment'],
                tweet_link=row['tweet_link']
            ))
        db.session.commit()

        # Simpan ke CSV (opsional)
        df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'hasil_klasifikasi.csv'), index=False)
        session['model_done'] = True
        flash('Klasifikasi selesai dan data disimpan.', 'success')
        return redirect(url_for('visualisasi'))

    return render_template('model.html', active_page='model')

@app.route('/visualisasi')
def visualisasi():
    if not session.get('model_done'):
        flash('Selesaikan tahap klasifikasi terlebih dahulu.', 'warning')
        return redirect(url_for('model'))

    data = DatasetKlasifikasi.query.all()
    df = pd.DataFrame([(d.tweet, d.sentiment) for d in data], columns=['tweet', 'sentiment'])

    # Hitung jumlah sentimen
    sentiment_counts = df['sentiment'].value_counts()
    total = sentiment_counts.sum()

    sentiment_labels = {
        'positive': 'Positif',
        'neutral': 'Netral',
        'negative': 'Negatif'
    }

    # Teks ringkasan
    summary = (
        f"Dari {total} tweet, "
        f"{sentiment_counts.get('positive', 0)} positif, "
        f"{sentiment_counts.get('neutral', 0)} netral, "
        f"{sentiment_counts.get('negative', 0)} negatif."
    )

    # 🔄 Ganti index label sentiment_counts ke label bahasa Indonesia
    sentiment_counts.index = sentiment_counts.index.map(lambda x: sentiment_labels.get(x, x))

    # Pie chart
    plt.figure(figsize=(6, 6))
    sentiment_counts.plot.pie(autopct='%1.1f%%', colors=["green", "grey", "red"])
    plt.ylabel('')
    plt.title("Distribusi Sentimen")
    plt.savefig("static/pie_chart.png")
    plt.close()

    # Ganti kolom baru untuk label Indonesia di bar chart
    df['sentiment_id'] = df['sentiment'].map(sentiment_labels)

    # Bar chart
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="sentiment_id", order=["Positif", "Netral", "Negatif"],
                palette={"Positif": "green", "Netral": "grey", "Negatif": "red"})
    plt.title("Jumlah Tweet per Sentimen")
    plt.xlabel("Sentimen")  # Ganti label sumbu X
    plt.ylabel("Total")     # Ganti label sumbu Y
    plt.savefig("static/bar_chart.png")
    plt.close()

    # Wordcloud umum
    all_text = " ".join(df["tweet"].astype(str))
    WordCloud(width=800, height=400, background_color='white').generate(all_text).to_file("static/wordcloud_all.png")

    # Wordcloud positif
    pos_text = " ".join(df[df["sentiment"] == "positive"]["tweet"])
    WordCloud(width=800, height=400, background_color='white').generate(pos_text).to_file("static/wordcloud_positive.png")

    # Wordcloud negatif
    neg_text = " ".join(df[df["sentiment"] == "negative"]["tweet"])
    WordCloud(width=800, height=400, background_color='white').generate(neg_text).to_file("static/wordcloud_negative.png")

    return render_template('visualisasi.html', active_page='visualisasi', data=data, summary=summary)

@app.route('/reset', methods=['POST'])
def reset():
    session.clear()  # hanya menghapus semua data sesi user (upload_done, preprocessing_done, dll)
    flash('Sesi berhasil direset. Anda dapat memulai kembali dari awal.', 'success')
    return redirect(url_for('home'))



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
