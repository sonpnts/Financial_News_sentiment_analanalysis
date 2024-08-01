from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from googletrans import Translator


app = Flask(__name__)

# Khởi tạo mô hình và tokenizer
# model_name = "ProsusAI/finbert"
model = AutoModelForSequenceClassification.from_pretrained('sentiment_analysis_accu-81')
tokenizer = AutoTokenizer.from_pretrained('sentiment_analysis_accu-81')


def translate_text(text):
    translator = Translator()
    translated = translator.translate(text, src='en', dest='vi')
    return translated.text
def fetch_news():
    url = 'https://news.google.com/rss/search?q=finance&hl=en-US&gl=US&ceid=US:en'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'xml')
    articles = soup.find_all('item')
    news_items = []

    for article in articles:
        news_item = {
            'title': article.title.text,
            'link': article.link.text,
            'pubDate': article.pubDate.text
        }
        news_items.append(news_item)

    return news_items

def clean_text(sentence):
    words = sentence.split()
    mytokens = [word for word in words]
    return ' '.join(mytokens).strip()

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = F.softmax(outputs.logits, dim=-1).tolist()
    return {
        "positive": probabilities[0][2],
        "negative": probabilities[0][0],
        "neutral": probabilities[0][1]
    }

def processing(input_text):
    cleaned_text = clean_text(input_text)
    prediction = predict(cleaned_text)
    return prediction

@app.route('/')
def load_rss():
    news_items = fetch_news()
    return render_template('rss.html', news_items=news_items)

@app.route('/analyze_article')
def analyze_article():
    title = request.args.get('title')
    link = request.args.get('link')
    prediction = processing(title)
    translated_title = translate_text(title)
    return render_template('analysis.html', title=title,titlevn=translated_title, prediction=prediction, link=link)

if __name__ == '__main__':
    app.run(debug=True)
