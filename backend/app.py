from flask import Flask, request, jsonify
from flask_cors import CORS
import os, re, json
import pandas as pd
import numpy as np
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

MODEL_DIR = 'models'
LR_MODEL_PATH = os.path.join(MODEL_DIR, 'lr_embeddings.pkl')

VALID_CATEGORIES = [
    'Fragrance',
    'Grocery & Gourmet Foods', 
    'Hair Care',
    'Other',
    'Personal Care'
]

MONGO_URI = os.getenv('ATLAS_URI')
mongo_client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = mongo_client['ecommerce_db']
products_coll = db['products']

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = 'prod-search'
pinecone_index = pc.Index(INDEX_NAME)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
lr_model = joblib.load(LR_MODEL_PATH)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text_for_embeddings(text):
    return clean_text(text)

def predict_category(query: str) -> str:
    processed_query = preprocess_text_for_embeddings(query)
    query_embedding = embed_model.encode([processed_query])
    prediction = lr_model.predict(query_embedding)[0]
    return prediction

def predict_with_confidence(query: str) -> tuple:
    processed_query = preprocess_text_for_embeddings(query)
    query_embedding = embed_model.encode([processed_query])
    prediction = lr_model.predict(query_embedding)[0]
    probabilities = lr_model.predict_proba(query_embedding)[0]
    confidence = max(probabilities)
    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_predictions = [(lr_model.classes_[i], probabilities[i]) for i in top_indices]
    return prediction, confidence, top_predictions

def parse_price_filter(query: str):
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

def generate_response_gemini(query: str, products: list, category: str = None, confidence: float = None):
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
    
    confidence_text = ""
    if confidence and confidence > 0.8:
        confidence_text = "I'm confident these are exactly what you're looking for!"
    elif confidence and confidence > 0.6:
        confidence_text = "These should be good matches for your needs."
    
    prompt = f"""
    You are a helpful e-commerce assistant. 
    The user searched for: "{query}"
    Category identified: {category if category else 'General'}
    
    Here are the top results:
    {product_info[0]}
    {product_info[1] if len(product_info) > 1 else ''}
    {product_info[2] if len(product_info) > 2 else ''}
    
    Write a natural-sounding response that:
    1. Acknowledges what they searched for
    2. Mentions we found {len(products)} products {brand_text} {price_text}
    3. {confidence_text}
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

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'status': 'error', 'message': 'Query is required'}), 400
    
    try:
        category, confidence, top_predictions = predict_with_confidence(query)
        
        if confidence < 0.5:
            return jsonify({
                'status': 'uncertain',
                'message': 'I\'m not sure about the category. Here are my top guesses:',
                'top_predictions': [{'category': cat, 'confidence': float(prob)} 
                                  for cat, prob in top_predictions],
                'category': category
            })
        
        if category == "Other":
            return jsonify({
                'status': 'not_found',
                'message': 'We currently don\'t carry items in that category. Check back soon!',
                'category': category,
                'confidence': float(confidence)
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
            
            nl_response = generate_response_gemini(query, products, category, confidence)
            
            return jsonify({
                'status': 'success',
                'category': category,
                'confidence': float(confidence),
                'products': serializable_products,
                'price_filter': search_results.get('price_filter', {}),
                'response': nl_response,
                'top_predictions': [{'category': cat, 'confidence': float(prob)} 
                                  for cat, prob in top_predictions[:3]]
            })
        else:
            return jsonify({
                'status': 'not_found',
                'message': search_results.get('message', 'No products found'),
                'category': category,
                'confidence': float(confidence)
            })
            
    except Exception as e:
        print(f"Error processing search: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predict-category', methods=['POST'])
def predict_category_endpoint():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'status': 'error', 'message': 'Query is required'}), 400
    
    try:
        category, confidence, top_predictions = predict_with_confidence(query)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'predicted_category': category,
            'confidence': float(confidence),
            'top_predictions': [{'category': cat, 'confidence': float(prob)} 
                              for cat, prob in top_predictions],
            'valid_categories': VALID_CATEGORIES
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'timestamp': time.time(),
        'valid_categories': VALID_CATEGORIES,
        'model_type': 'sentence_transformers'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)