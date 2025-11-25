import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="ProcureAI - Commercial Procurement Assistant",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class ProcurementAI:
    def __init__(self):
        self.suppliers = self.generate_suppliers()
        self.price_data = self.generate_price_data()
    
    def generate_suppliers(self):
        suppliers = []
        for i in range(30):
            suppliers.append({
                'id': f'SUP{i+1:03d}',
                'name': f'Supplier {i+1}',
                'category': random.choice(['Electronics', 'Materials', 'Logistics', 'Services']),
                'rating': round(random.uniform(3.0, 5.0), 1),
                'delivery_days': random.randint(2, 15),
                'risk': random.choice(['Low', 'Medium', 'High']),
                'last_order': datetime.now() - timedelta(days=random.randint(1, 180))
            })
        return pd.DataFrame(suppliers)
    
    def generate_price_data(self):
        products = ['Laptop', 'Steel', 'Plastic', 'Shipping', 'Packaging']
        data = []
        for product in products:
            base_price = random.uniform(50, 500)
            for month in range(12):
                data.append({
                    'month': month + 1,
                    'product': product,
                    'price': base_price * (1 + random.uniform(-0.2, 0.2)),
                    'demand': random.randint(100, 1000)
                })
        return pd.DataFrame(data)
    
    def predict_prices(self, product, months=6):
        product_data = self.price_data[self.price_data['product'] == product]
        if len(product_data) == 0:
            return []
        
        current_price = product_data['price'].iloc[-1]
        trend = random.uniform(-0.1, 0.1)
        
        predictions = []
        for i in range(months):
            future_price = current_price * (1 + trend * (i + 1))
            predictions.append({
                'month': f'Month {i+1}',
                'price': future_price,
                'trend': 'up' if future_price > current_price else 'down'
            })
        return predictions
    
    def find_optimal_order(self, product, budget, current_stock):
        predictions = self.predict_prices(product, 3)
        if not predictions:
            return None
        
        best_price = min(p['price'] for p in predictions)
        best_month = [i for i, p in enumerate(predictions) if p['price'] == best_price][0] + 1
        
        max_order = budget / best_price
        optimal_qty = min(max_order, current_stock * 1.5)
        
        return {
            'quantity': int(optimal_qty),
            'month': best_month,
            'unit_price': best_price,
            'total_cost': optimal_qty * best_price,
            'savings': (predictions[0]['price'] - best_price) * optimal_qty
        }

def main():
    ai = ProcurementAI()
    
    st.markdown('<div class="main-header">🚀 ProcureAI</div>', unsafe_allow_html=True)
    st.markdown('**Commercial AI Procurement Assistant**')
    
    # Sidebar navigation
    st.sidebar.title("AI Modules")
    page = st.sidebar.radio("Navigate to:", 
                           ["Dashboard", "Price Predictions", "Supplier Analytics", "Order Optimization"])
    
    if page == "Dashboard":
        show_dashboard(ai)
    elif page == "Price Predictions":
        show_predictions(ai)
    elif page == "Supplier Analytics":
        show_suppliers(ai)
    else:
        show_optimization(ai)

def show_dashboard(ai):
    st.header("📊 Procurement Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_rating = ai.suppliers['rating'].mean()
        st.metric("Avg Supplier Rating", f"{avg_rating:.1f}/5.0")
    
    with col2:
        low_risk = len(ai.suppliers[ai.suppliers['risk'] == 'Low'])
        st.metric("Low Risk Suppliers", low_risk)
    
    with col3:
        avg_delivery = ai.suppliers['delivery_days'].mean()
        st.metric("Avg Delivery Time", f"{avg_delivery:.1f} days")
    
    with col4:
        products = len(ai.price_data['product'].unique())
        st.metric("Tracked Products", products)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Supplier Risk Distribution")
        risk_counts = ai.suppliers['risk'].value_counts()
        st.bar_chart(risk_counts)
    
    with col2:
        st.subheader("Current Prices")
        current_prices = ai.price_data.groupby('product')['price'].last()
        st.bar_chart(current_prices)
    
    # AI Recommendations
    st.subheader("🤖 AI Recommendations")
    
    rec1, rec2, rec3 = st.columns(3)
    
    with rec1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**💡 Cost Saving Opportunity**")
        st.markdown("Bulk order plastics next month for 12% savings")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with rec2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**⚡ High-Performance Supplier**")
        st.markdown("Supplier 15: 4.8 rating, 3-day delivery")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with rec3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**📈 Market Trend**")
        st.markdown("Electronics prices trending down 8% next quarter")
        st.markdown("</div>", unsafe_allow_html=True)

def show_predictions(ai):
    st.header("🔮 AI Price Predictions")
    
    product = st.selectbox("Select Product", ai.price_data['product'].unique())
    months = st.slider("Forecast Period (months)", 3, 12, 6)
    
    if st.button("Generate AI Prediction", type="primary"):
        predictions = ai.predict_prices(product, months)
        
        if predictions:
            st.success("AI Analysis Complete!")
            
            # Show predictions
            st.subheader(f"Price Forecast for {product}")
            for pred in predictions:
                trend_icon = "📈" if pred['trend'] == 'up' else "📉"
                st.write(f"{pred['month']}: ${pred['price']:.2f} {trend_icon}")
            
            # Chart
            pred_df = pd.DataFrame(predictions)
            st.line_chart(pred_df.set_index('month')['price'])
            
            # Historical data
            st.subheader("Historical Data")
            hist_data = ai.price_data[ai.price_data['product'] == product]
            st.line_chart(hist_data.set_index('month')['price'])

def show_suppliers(ai):
    st.header("🏭 Supplier Analytics")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        min_rating = st.slider("Minimum Rating", 3.0, 5.0, 4.0)
    
    with col2:
        max_delivery = st.slider("Max Delivery Days", 2, 20, 10)
    
    # Filter suppliers
    filtered = ai.suppliers[
        (ai.suppliers['rating'] >= min_rating) & 
        (ai.suppliers['delivery_days'] <= max_delivery)
    ]
    
    st.subheader(f"Filtered Suppliers ({len(filtered)})")
    st.dataframe(filtered, use_container_width=True)
    
    # Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance by Category")
        category_avg = filtered.groupby('category')['rating'].mean()
        st.bar_chart(category_avg)
    
    with col2:
        st.subheader("Risk Analysis")
        risk_analysis = filtered['risk'].value_counts()
        st.bar_chart(risk_analysis)

def show_optimization(ai):
    st.header("📦 Order Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        product = st.selectbox("Product:", ai.price_data['product'].unique(), key="opt_product")
        budget = st.number_input("Budget ($):", 1000, 1000000, 50000)
        inventory = st.number_input("Current Inventory:", 0, 10000, 100)
    
    if st.button("Calculate Optimal Order", type="primary"):
        result = ai.find_optimal_order(product, budget, inventory)
        
        if result:
            st.success("🎯 AI Recommendation Generated!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Order Quantity", f"{result['quantity']} units")
                st.metric("Best Timing", f"Month {result['month']}")
            
            with col2:
                st.metric("Unit Price", f"${result['unit_price']:.2f}")
                st.metric("Total Cost", f"${result['total_cost']:.0f}")
            
            with col3:
                st.metric("Potential Savings", f"${result['savings']:.0f}")
                st.metric("ROI", f"{(result['savings']/result['total_cost']*100):.1f}%")

if __name__ == "__main__":
    main()
