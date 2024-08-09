import json
import urllib
from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from googletrans import Translator
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

app = Flask(__name__)

# Khởi tạo mô hình và tokenizer
# model_name = "ProsusAI/finbert"
model = AutoModelForSequenceClassification.from_pretrained('sentiment_analysis_accu-81')
tokenizer = AutoTokenizer.from_pretrained('sentiment_analysis_accu-81')

username = urllib.parse.quote_plus('son')
password = urllib.parse.quote_plus('lij2UqPtF0RQ7anx')

# Tạo URI kết nối MongoDB với thông tin mã hóa
uri = f"mongodb+srv://{username}:{password}@sonpnts.akwoo40.mongodb.net/?retryWrites=true&w=majority&appName=sonpnts"
# uri = f"mongodb+srv://sonpnts:Son1010@@sonpnts.akwoo40.mongodb.net/?retryWrites=true&w=majority&appName=sonpnts"

client = MongoClient(uri, server_api=ServerApi('1'))

db = client['sentiment_analysis']
collection = db['news']

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

@app.route('/analyze_article', methods=['POST'])
def analyze_article():
    item = request.get_json()
    title = item['title']
    link = item['link']
    prediction = processing(title)
    document = {
        'title': title,
        'link': link,
        'prediction': prediction,
        'pubDate': item['pubDate'],
        'positive': prediction['positive'],
        'negative': prediction['negative'],
        'neutral': prediction['neutral']
    }

    # Lưu tài liệu vào MongoDB
    collection.insert_one(document)
    return jsonify({'title': title, 'link': link, 'prediction': prediction})

@app.route('/analysis')
def analysis():
    title = request.args.get('title')
    link = request.args.get('link')
    prediction_url = request.args.get('prediction')
    prediction = json.loads(prediction_url) if prediction_url else {}
    translated_title = translate_text(title)
    return render_template('analysis.html', title=title,titlevn=translated_title, prediction=prediction, link=link)

@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    item = request.get_json()
    url = item['url']
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)

    # Fetch and process the content from the URL
    # response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.title.string if soup.title else 'No title found'
    content = soup.get_text()

    # Analyze the content
    prediction = processing(content)
    # document = {
    #     'title': title,
    #     'link': url,
    #     'prediction': prediction,
    #     'positive': prediction['positive'],
    #     'negative': prediction['negative'],
    #     'neutral': prediction['neutral']
    # }
    #
    # # Save to MongoDB
    # collection.insert_one(document)
    print(title, url, prediction)
    return jsonify({'title': title, 'link': url, 'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
