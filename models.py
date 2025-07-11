from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class DatasetMentah(db.Model):
    __tablename__ = 'dataset_mentah'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    tweet = db.Column(db.Text)
    waktu = db.Column(db.String(50))
    tweet_link = db.Column(db.String(255))  # kolom tweet_link baru

    def __repr__(self):
        return f'<DatasetMentah id={self.id} username={self.username}>'

class DatasetPreprocessing(db.Model):
    __tablename__ = 'dataset_preprocessing'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    tweet = db.Column(db.Text)
    text_clean = db.Column(db.Text)
    waktu = db.Column(db.String(50))
    tweet_link = db.Column(db.String(255))  # kolom tweet_link baru

    def __repr__(self):
        return f'<DatasetPreprocessing id={self.id} username={self.username}>'

class DatasetKlasifikasi(db.Model):
    __tablename__ = 'dataset_klasifikasi'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    tweet = db.Column(db.Text)
    text_clean = db.Column(db.Text)
    waktu = db.Column(db.String(50))
    sentiment = db.Column(db.String(50))
    tweet_link = db.Column(db.String(255))  # kolom tweet_link baru

    def __repr__(self):
        return f'<DatasetKlasifikasi id={self.id} sentiment={self.sentiment}>'
