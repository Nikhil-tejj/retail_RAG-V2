from flask import Flask, request, jsonify
from flask_cors import CORS
import os, re, json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
import certifi
from pymongo import MongoClient
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import joblib
import google.generativeai as genai
import time

app = Flask(__name__)
CORS(app)  

load_dotenv()

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

MODEL_DIR = 'models'
TFIDF_PATH = os.path.join(MODEL_DIR, 'tfidf_preprocessed.pkl')
LR_PATH = os.path.join(MODEL_DIR, 'lr_preprocessed.pkl')

MONGO_URI = os.getenv('ATLAS_URI')
mongo_client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = mongo_client['ecommerce_db']
products_coll = db['products']

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = 'prod-search'
pinecone_index = pc.Index(INDEX_NAME)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

def clean_text(text):
    """Basic text cleaning"""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    """Remove stopwords"""
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

def lemmatize_text(text):
    """Lemmatize text"""
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def preprocess_text(text):
    """Full preprocessing pipeline"""
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

def load_preprocessed_model():
    """Load the preprocessed model components"""
    tfidf = joblib.load(TFIDF_PATH)
    lr = joblib.load(LR_PATH)
    return tfidf, lr

def predict_category(query: str) -> str:
    """Predict category with preprocessing"""
    processed_query = preprocess_text(query)
    
    tfidf, lr = load_preprocessed_model()
    X = tfidf.transform([processed_query])
    return lr.predict(X)[0]

def parse_price_filter(query: str):
    """Parse price filters from query"""
    price_filter = {}
    
    lower_patterns = [
        r'(under|below|less\s*than)\s*[₹$]?\s*(\d+)',
        r'under\s*(\d+)',
        r'below\s*(\d+)'
    ]
    
    for pattern in lower_patterns:
        m = re.search(pattern, query, flags=re.I)
        if m:
            try:
                price = float([g for g in m.groups() if g and g.isdigit()][-1])
                price_filter['max_price'] = price
                break
            except (ValueError, IndexError):
                continue
    
    upper_patterns = [
        r'(above|over|more\s*than|greater\s*than)\s*[₹$]?\s*(\d+)',
        r'above\s*(\d+)',
        r'over\s*(\d+)',
        r'minimum\s*[₹$]?\s*(\d+)',
        r'at\s*least\s*[₹$]?\s*(\d+)'
    ]
    
    for pattern in upper_patterns:
        m = re.search(pattern, query, flags=re.I)
        if m:
            try:
                price = float([g for g in m.groups() if g and g.isdigit()][-1])
                price_filter['min_price'] = price
                break
            except (ValueError, IndexError):
                continue
    
    range_patterns = [
        r'between\s*[₹$]?\s*(\d+)\s*(?:and|to|-)\s*[₹$]?\s*(\d+)',
        r'(\d+)\s*(?:to|-)\s*(\d+)',
        r'from\s*[₹$]?\s*(\d+)\s*(?:to|-)\s*[₹$]?\s*(\d+)'
    ]
    
    for pattern in range_patterns:
        m = re.search(pattern, query, flags=re.I)
        if m:
            try:
                min_price = float(m.group(1))
                max_price = float(m.group(2))
                if min_price <= max_price:
                    price_filter['min_price'] = min_price
                    price_filter['max_price'] = max_price
                else:
                    price_filter['min_price'] = max_price
                    price_filter['max_price'] = min_price
                break
            except (ValueError, IndexError):
                continue
    
    return price_filter if price_filter else None

def search_products(query: str, top_k: int = 10):
    """Search for products using RAG"""
    q_emb = embed_model.encode(query).tolist()
    res = pinecone_index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    matches = res.get('matches', [])
    if not matches:
        return {'status': 'not_found', 'message': 'No vector matches.'}

    ids = [m['id'] for m in matches]
    price_filter = parse_price_filter(query)

    mongo_filter = {'unique_id': {'$in': ids}}
    if price_filter:
        price_query = {}
        if 'min_price' in price_filter:
            price_query['$gte'] = price_filter['min_price']
        if 'max_price' in price_filter:
            price_query['$lte'] = price_filter['max_price']
        if price_query:
            mongo_filter['price'] = price_query
    
    docs = list(products_coll.find(mongo_filter))
    if not docs:
        filter_msg = ""
        if price_filter:
            if 'min_price' in price_filter and 'max_price' in price_filter:
                filter_msg = f" with price between ₹{price_filter['min_price']} and ₹{price_filter['max_price']}"
            elif 'min_price' in price_filter:
                filter_msg = f" with price above ₹{price_filter['min_price']}"
            elif 'max_price' in price_filter:
                filter_msg = f" with price under ₹{price_filter['max_price']}"
        return {'status': 'not_found', 'message': f'No products matched after filtering{filter_msg}.'}

    score_map = {m['id']: m.get('score', 0) for m in matches}
    docs.sort(key=lambda d: score_map.get(str(d.get('unique_id')), 0), reverse=True)
    
    result = {'status': 'success', 'results': docs}
    if price_filter:
        result['price_filter'] = price_filter
    
    return result

def generate_response_gemini(query: str, products: list):
    """Generate natural language response using Gemini"""
    if not products:
        return 'I could not find matching products right now.'
    
    top = products[:3] 
    brands = list({str(p.get('brand','')).strip() for p in top if p.get('brand') and str(p.get('brand','')).strip()})
    pr = [float(p.get('price', 0)) for p in top if isinstance(p.get('price', 0), (int, float)) and p.get('price', 0) > 0]
    pr_min = min(pr) if pr else None
    pr_max = max(pr) if pr else None
    
    product_info = []
    for i, p in enumerate(top, 1):
        title = str(p.get('title', 'Product')).strip()
        if len(title) > 50:
            title = title[:47] + "..."
        brand = str(p.get('brand', 'Unknown')).strip()
        price = p.get('price', 0)
        product_info.append(f"{i}. {title} by {brand} (₹{price})")
    
    brand_text = f"from brands like {', '.join(brands[:2])}" if brands else ""
    price_text = ""
    if pr_min and pr_max:
        if pr_min == pr_max:
            price_text = f"priced at ₹{pr_min}"
        else:
            price_text = f"ranging from ₹{pr_min} to ₹{pr_max}"
    
    prompt = f"""
    You are a helpful e-commerce assistant. 
    The user searched for: "{query}"
    
    Here are the top results:
    {product_info[0]}
    {product_info[1] if len(product_info) > 1 else ''}
    {product_info[2] if len(product_info) > 2 else ''}
    
    Write a natural-sounding response that:
    1. Acknowledges what they searched for
    2. Mentions we found {len(products)} products {brand_text} {price_text}
    3. Convinces them these are good matches in 2-3 sentences
    4. Keep it under 100 words and friendly
    """
    
    try:
        response = gemini_model.generate_content(prompt).text
        return response.strip()
    except Exception as e:
        print(f"Gemini generation error: {e}")
        brand_mention = f" from {brands[0]}" if brands else ""
        price_mention = f" under ₹{pr_max}" if pr_max else ""
        return f"I found {len(products)} great products matching your search{brand_mention}{price_mention}. These options should meet your needs perfectly."

# API endpoints
@app.route('/api/search', methods=['POST'])
def search():
    """Main search endpoint that combines category prediction and RAG search"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'status': 'error', 'message': 'Query is required'}), 400
    
    try:
        category = predict_category(query)
        
        if category == "Other":
            return jsonify({
                'status': 'not_found',
                'message': 'We currently don’t carry items in that category. Check back soon!',
                'category': category
            })
        
        search_results = search_products(query, top_k=10)
        
        if search_results.get('status') == 'success':
            products = search_results['results']
            
            serializable_products = []
            for p in products:
                product = {
                    'title': p.get('title', ''),
                    'description': p.get('description', ''),
                    'brand': p.get('brand', ''),
                    'price': p.get('price', 0),
                    'category': p.get('category', ''),
                    'image_urls': p.get('image_urls', [])
                }
                serializable_products.append(product)
            
            nl_response = generate_response_gemini(query, products)
            
            return jsonify({
                'status': 'success',
                'category': category,
                'products': serializable_products,
                'price_filter': search_results.get('price_filter', {}),
                'response': nl_response
            })
        else:
            return jsonify({
                'status': 'not_found',
                'message': search_results.get('message', 'No products found'),
                'category': category
            })
            
    except Exception as e:
        print(f"Error processing search: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)