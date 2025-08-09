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
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:5000/api"  

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

def main():
    st.title("🛍️ RetailSearch AI")
    st.markdown("### Find the perfect products with AI-powered search")

    query = st.text_input(
        "What are you looking for today?", 
        placeholder="e.g., 'moisturizing body wash', 'shampoo for dry hair', 'detergent under 200'"
    )
    search_button = st.button("Search", type="primary")

    if search_button and query:
        with st.spinner("Searching for products..."):
            results = search_products(query)
            
            if results and results.get("status") == "success":
                st.markdown("### 🤖 AI Assistant")
                st.markdown(f"<div class='ai-response'>{results['response']}</div>", unsafe_allow_html=True)
                
                st.markdown("### 🔍 Search Results")
                st.markdown(f"**Category: {results['category']}**")
                
                if 'price_filter' in results and results['price_filter']:
                    price_filter = results['price_filter']
                    filter_text = []
                    if 'min_price' in price_filter:
                        filter_text.append(f"Min: ₹{price_filter['min_price']}")
                    if 'max_price' in price_filter:
                        filter_text.append(f"Max: ₹{price_filter['max_price']}")
                    if filter_text:
                        st.markdown(f"**Price Filter: {', '.join(filter_text)}**")
                
                products = results['products']
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
                                    
                                st.markdown("</div>", unsafe_allow_html=True)
            
            elif results and results.get("status") == "not_found":
                st.warning(results.get("message", "No products found matching your search."))
                if "category" in results:
                    st.info(f"Product Category: {results['category']}")
            
            else:
                st.error("An error occurred while processing your search.")

    with st.sidebar:
        st.title("About")
        st.markdown("""
        This search tool uses AI to understand your natural language requests
        and find the most relevant products from our catalog.

        **Available Categories:**
        - Personal Care
        - Detergents & Dishwash
        - Fragrance
        - Grocery & Gourmet Foods
        - Hair Care
        - Other

        **Features:**
        - Natural language search
        - Category recognition
        - AI-powered responses
        - Price filtering (try "under 500" or "between 200 and 800")
        """)

    st.markdown("---")
    st.markdown("RetailSearch AI © 2023 - Powered by RAG + LLM😁")

if __name__ == "__main__":
    main()