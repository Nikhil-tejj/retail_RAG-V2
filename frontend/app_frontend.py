import streamlit as st
import requests
import json
from PIL import Image
import io
from urllib.request import urlopen

st.set_page_config(
    page_title="RetailSearch AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .product-card {
        border: 1px solid #eee;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .product-title {
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 5px;
    }
    .product-brand {
        color: #666;
        font-size: 0.9rem;
    }
    .product-price {
        font-weight: bold;
        color: #e63946;
        font-size: 1.1rem;
    }
    .ai-response {
        background: linear-gradient(135deg, #e6f2ff, #b3d9ff);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #7cb9ff;
        color: #0a5299;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    .confidence-indicator {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 10px 0;
    }
    .confidence-high {
        border-left-color: #28a745;
        background: #f1f8e9;
    }
    .confidence-medium {
        border-left-color: #ffc107;
        background: #fffbf0;
    }
    .confidence-low {
        border-left-color: #dc3545;
        background: #fff5f5;
    }
    .category-test-result {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #6c757d;
        color: #495057;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    .prediction-badge {
        display: inline-block;
        background: #e9ecef;
        padding: 3px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.8rem;
        color: #495057;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:7860/api"

def search_products(query):
    """Send search query to backend API and return results."""
    try:
        response = requests.post(
            f"{API_URL}/search",
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return {"status": "error", "error": str(e)}

def test_category_prediction(query):
    """Test category prediction endpoint."""
    try:
        response = requests.post(
            f"{API_URL}/predict-category",
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}

def get_confidence_class(confidence):
    """Get CSS class based on confidence level."""
    if confidence > 0.8:
        return "confidence-high"
    elif confidence > 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

def display_confidence_info(category, confidence, top_predictions):
    """Display confidence and prediction information."""
    confidence_class = get_confidence_class(confidence)
    confidence_pct = confidence * 100
    
    st.markdown(f"""
    <div class='confidence-indicator {confidence_class}'>
        <strong>🎯 Category Prediction:</strong> {category}<br>
        <strong>🔍 Confidence:</strong> {confidence_pct:.1f}%<br>
        <strong>📊 Top Predictions:</strong><br>
        {''.join([f'<span class="prediction-badge">{pred["category"]} ({pred["confidence"]*100:.0f}%)</span>' for pred in top_predictions[:3]])}
    </div>
    """, unsafe_allow_html=True)

def display_category_test_result(category, confidence, top_predictions):
    """Display category test result with different styling."""
    confidence_pct = confidence * 100
    
    st.markdown(f"""
    <div class='category-test-result'>
        <strong>🧪 Category Test Result</strong><br>
        <strong>🎯 Predicted Category:</strong> {category}<br>
        <strong>🔍 Confidence:</strong> {confidence_pct:.1f}%<br>
        <strong>📊 All Predictions:</strong><br>
        {''.join([f'<span class="prediction-badge">{pred["category"]} ({pred["confidence"]*100:.0f}%)</span>' for pred in top_predictions[:3]])}
    </div>
    """, unsafe_allow_html=True)

def main():
    st.title("🛍️ RetailSearch AI")
    st.markdown("### Find the perfect products with AI-powered search")

    query = st.text_input(
        "What are you looking for today?", 
        placeholder="e.g., 'moisturizing body wash', 'organic shampoo', 'vanilla perfume under 500'"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_button = st.button("Search", type="primary")
    with col2:
        test_button = st.button("🧪 Test Category", help="Test category prediction only")

    if test_button and query:
        with st.spinner("Testing category prediction..."):
            result = test_category_prediction(query)
            if result.get("status") == "success":
                display_category_test_result(
                    result["predicted_category"], 
                    result["confidence"], 
                    result["top_predictions"]
                )
            else:
                st.error("Error testing category prediction")

    if search_button and query:
        with st.spinner("Searching for products..."):
            results = search_products(query)
            
            if results and results.get("status") == "success":
                display_confidence_info(
                    results["category"],
                    results["confidence"], 
                    results["top_predictions"]
                )
                
                st.markdown("### 🤖 AI Assistant")
                st.markdown(f"<div class='ai-response'>{results['response']}</div>", unsafe_allow_html=True)
                
                st.markdown("### 🔍 Search Results")
                
                if 'price_filter' in results and results['price_filter']:
                    price_filter = results['price_filter']
                    filter_text = []
                    if 'min_price' in price_filter:
                        filter_text.append(f"Min: ₹{price_filter['min_price']}")
                    if 'max_price' in price_filter:
                        filter_text.append(f"Max: ₹{price_filter['max_price']}")
                    if filter_text:
                        st.markdown(f"**💰 Price Filter Applied: {', '.join(filter_text)}**")
                
                products = results['products']
                st.markdown(f"**📦 Found {len(products)} products**")
                
                cols = 3 
                
                for i in range(0, len(products), cols):
                    row = st.columns(cols)
                    for j in range(cols):
                        if i + j < len(products):
                            product = products[i + j]
                            with row[j]:
                                st.markdown(f"<div class='product-card'>", unsafe_allow_html=True)
                                
                                if product.get('image_urls') and len(product['image_urls']) > 0:
                                    img_url = product['image_urls'][0]
                                    if img_url.strip():
                                        try:
                                            st.image(img_url, use_container_width=True)
                                        except:
                                            st.info("Image not available")
                                
                                st.markdown(f"<div class='product-title'>{product['title'][:50]}{'...' if len(product['title']) > 50 else ''}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='product-brand'>Brand: {product['brand']}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='product-price'>₹{product['price']:.2f}</div>", unsafe_allow_html=True)
                                
                                with st.expander("Details"):
                                    st.write(product['description'][:200] + '...' if len(product['description']) > 200 else product['description'])
                                    if product.get('category'):
                                        st.write(f"**Category:** {product['category']}")
                                    
                                st.markdown("</div>", unsafe_allow_html=True)
            
            elif results and results.get("status") == "uncertain":
                st.warning("🤔 " + results.get("message", "I'm not sure about the category."))
                if results.get("top_predictions"):
                    st.markdown("**My best guesses:**")
                    for pred in results["top_predictions"]:
                        st.markdown(f"- {pred['category']} ({pred['confidence']*100:.0f}% confidence)")
                st.info("Try being more specific or use different keywords.")
            
            elif results and results.get("status") == "not_found":
                st.warning("😔 " + results.get("message", "No products found matching your search."))
                if "category" in results:
                    st.info(f"**Detected Category:** {results['category']}")
                    if results.get("confidence"):
                        st.info(f"**Confidence:** {results['confidence']*100:.1f}%")
            
            else:
                st.error("❌ An error occurred while processing your search.")
                if results.get("error"):
                    st.code(results["error"])

    with st.sidebar:
        st.title("About")
        st.markdown("""
        This search tool uses **Sentence Transformers + AI** to understand your natural language requests
        and find the most relevant products from our catalog.

        **🏷️ Available Categories:**
        - Personal Care
        - Fragrance  
        - Grocery & Gourmet Foods
        - Hair Care
        - Other

        **✨ Features:**
        - **Confidence Scores** - See how certain the AI is
        - **Category Testing** - Test predictions without searching
        - **Improved Accuracy** - 85%+ classification accuracy
        - **Semantic Understanding** - Better synonym matching

        **🔍 Search Tips:**
        - Use natural language: "shampoo for dry hair"
        - Add price filters: "under 500" or "between 200-800"
        - Be specific: "organic moisturizer" vs just "cream"
        
        **🧠 Model Info:**
        - **Embeddings:** Sentence Transformers (384-dim)
        - **Classifier:** Logistic Regression
        - **Categories:** 5 main categories
        """)

    st.markdown("---")
    st.markdown("RetailSearch AI © 2025 - Powered by Sentence Transformers + RAG 🚀")

if __name__ == "__main__":
    main()