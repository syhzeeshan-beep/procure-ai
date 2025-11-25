import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="ProcureAI - Commercial Procurement Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .prediction-positive {
        color: #00d154;
        font-weight: bold;
    }
    .prediction-negative {
        color: #ff4b4b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class ProcurementAssistant:
    def __init__(self):
        self.supplier_data = self.generate_supplier_data()
        self.price_history = self.generate_price_history()
        self.demand_patterns = self.generate_demand_patterns()
    
    def generate_supplier_data(self):
        suppliers = []
        categories = ['Electronics', 'Raw Materials', 'Packaging', 'Logistics', 'Office Supplies']
        risk_levels = ['Low', 'Medium', 'High']
        
        for i in range(50):
            suppliers.append({
                'id': f'SUP{i+1:03d}',
                'name': f'Supplier {i+1}',
                'category': np.random.choice(categories),
                'reliability_score': np.random.uniform(3.0, 5.0),
                'delivery_time_days': np.random.randint(1, 30),
                'risk_level': np.random.choice(risk_levels, p=[0.6, 0.3, 0.1]),
                'cost_score': np.random.uniform(1.0, 5.0),
                'last_audit_date': datetime.now() - timedelta(days=np.random.randint(1, 365))
            })
        return pd.DataFrame(suppliers)
    
    def generate_price_history(self):
        dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='M')
        products = ['Laptop', 'Steel', 'Plastic', 'Cardboard', 'Shipping']
        
        data = []
        for date in dates:
            for product in products:
                base_price = np.random.uniform(100, 1000)
                trend = np.random.uniform(-0.1, 0.1)
                data.append({
                    'date': date,
                    'product': product,
                    'price': base_price * (1 + trend * ((date - dates[0]).days / 30)),
                    'volume': np.random.randint(100, 1000)
                })
        return pd.DataFrame(data)
    
    def generate_demand_patterns(self):
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        products = ['Laptop', 'Steel', 'Plastic', 'Cardboard', 'Shipping']
        
        data = []
        for month in months:
            for product in products:
                seasonal_factor = 1 + 0.3 * np.sin(months.index(month) * np.pi / 6)
                data.append({
                    'month': month,
                    'product': product,
                    'demand': np.random.randint(500, 2000) * seasonal_factor,
                    'forecast_accuracy': np.random.uniform(0.8, 0.95)
                })
        return pd.DataFrame(data)
    
    def predict_price_trends(self, product, months=6):
        """AI-powered price trend prediction"""
        current_data = self.price_history[self.price_history['product'] == product]
        if len(current_data) == 0:
            return None
        
        # Simple trend analysis (in real app, use ML model)
        recent_prices = current_data.tail(3)['price'].values
        if len(recent_prices) >= 2:
            trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        else:
            trend = 0
        
        # Add some randomness for simulation
        trend += np.random.uniform(-0.15, 0.15)
        
        predictions = []
        current_price = recent_prices[-1] if len(recent_prices) > 0 else np.random.uniform(100, 1000)
        
        for i in range(months):
            predicted_price = current_price * (1 + trend * (i + 1))
            predictions.append({
                'month': i + 1,
                'predicted_price': predicted_price,
                'confidence': max(0.7, 1 - abs(trend) * 2)
            })
        
        return predictions
    
    def calculate_optimal_order(self, product, budget, current_inventory):
        """AI-powered optimal order calculation"""
        predictions = self.predict_price_trends(product, 3)
        if not predictions:
            return None
        
        # Simple optimization logic
        predicted_prices = [p['predicted_price'] for p in predictions]
        best_month = predicted_prices.index(min(predicted_prices))
        
        optimal_quantity = min(budget / predicted_prices[best_month], 
                             current_inventory * 2)  # Don't order more than 2x current inventory
        
        return {
            'optimal_quantity': optimal_quantity,
            'best_month': best_month + 1,
            'predicted_price': predicted_prices[best_month],
            'total_cost': optimal_quantity * predicted_prices[best_month],
            'savings_potential': (max(predicted_prices) - min(predicted_prices)) * optimal_quantity
        }

def main():
    # Initialize the procurement assistant
    proc_assistant = ProcurementAssistant()
    
    # Header
    st.markdown('<div class="main-header">🚀 ProcureAI</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 3rem;">Commercial AI Procurement Assistant</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3714/3714885.png", width=100)
        st.title("Navigation")
        
        menu_option = st.radio(
            "Select Module:",
            ["📊 Dashboard", "🔮 Price Predictions", "📦 Optimal Orders", 
             "🏭 Supplier Analysis", "📈 Demand Forecasting"]
        )
        
        st.markdown("---")
        st.subheader("Quick Actions")
        if st.button("🔄 Refresh Data"):
            st.rerun()
        
        st.markdown("---")
        st.info("""
        **ProcureAI** helps businesses:
        - Predict price trends
        - Optimize order quantities
        - Analyze supplier performance
        - Forecast demand patterns
        """)
    
    # Main content based on menu selection
    if menu_option == "📊 Dashboard":
        show_dashboard(proc_assistant)
    elif menu_option == "🔮 Price Predictions":
        show_price_predictions(proc_assistant)
    elif menu_option == "📦 Optimal Orders":
        show_optimal_orders(proc_assistant)
    elif menu_option == "🏭 Supplier Analysis":
        show_supplier_analysis(proc_assistant)
    elif menu_option == "📈 Demand Forecasting":
        show_demand_forecasting(proc_assistant)

def show_dashboard(assistant):
    st.markdown('<div class="sub-header">Procurement Dashboard</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_reliability = assistant.supplier_data['reliability_score'].mean()
        st.metric("Avg Supplier Reliability", f"{avg_reliability:.1f}/5.0", "0.2")
    
    with col2:
        low_risk_suppliers = len(assistant.supplier_data[assistant.supplier_data['risk_level'] == 'Low'])
        st.metric("Low Risk Suppliers", low_risk_suppliers, "5")
    
    with col3:
        avg_delivery = assistant.supplier_data['delivery_time_days'].mean()
        st.metric("Avg Delivery Time", f"{avg_delivery:.1f} days", "-2.1")
    
    with col4:
        total_products = len(assistant.price_history['product'].unique())
        st.metric("Tracked Products", total_products, "2")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Supplier Risk Distribution")
        risk_counts = assistant.supplier_data['risk_level'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                    color=risk_counts.index,
                    color_discrete_map={'Low':'green', 'Medium':'orange', 'High':'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Price Trends Overview")
        recent_prices = assistant.price_history.groupby('product')['price'].last()
        fig = px.bar(x=recent_prices.index, y=recent_prices.values,
                    title="Current Prices by Product")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent alerts
    st.subheader("AI Recommendations & Alerts")
    
    alert_col1, alert_col2, alert_col3 = st.columns(3)
    
    with alert_col1:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**📈 Price Drop Alert**")
            st.markdown("Steel prices predicted to drop 8% in next 2 months")
            st.markdown("</div>", unsafe_allow_html=True)
    
    with alert_col2:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**⚠️ Supplier Risk**")
            st.markdown("3 suppliers require immediate audit")
            st.markdown("</div>", unsafe_allow_html=True)
    
    with alert_col3:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**💰 Cost Saving**")
            st.markdown("Potential 15% savings on logistics")
            st.markdown("</div>", unsafe_allow_html=True)

def show_price_predictions(assistant):
    st.markdown('<div class="sub-header">AI Price Trend Predictions</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_product = st.selectbox(
            "Select Product:",
            assistant.price_history['product'].unique()
        )
        
        prediction_months = st.slider("Prediction Period (months):", 3, 12, 6)
        
        if st.button("Generate Price Forecast", type="primary"):
            predictions = assistant.predict_price_trends(selected_product, prediction_months)
            
            if predictions:
                # Display predictions
                st.subheader(f"Price Forecast for {selected_product}")
                
                for pred in predictions:
                    trend_icon = "📈" if pred['predicted_price'] > predictions[0]['predicted_price'] else "📉"
                    st.write(f"Month {pred['month']}: ${pred['predicted_price']:.2f} {trend_icon}")
                
                # Create prediction chart
                pred_df = pd.DataFrame(predictions)
                fig = px.line(pred_df, x='month', y='predicted_price',
                            title=f"{selected_product} Price Prediction",
                            markers=True)
                fig.update_traces(line=dict(color='#1f77b4', width=3))
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Show historical price data
        st.subheader("Historical Price Data")
        product_data = assistant.price_history[assistant.price_history['product'] == selected_product]
        fig = px.line(product_data, x='date', y='price',
                     title=f"Historical Prices for {selected_product}")
        st.plotly_chart(fig, use_container_width=True)

def show_optimal_orders(assistant):
    st.markdown('<div class="sub-header">AI-Powered Optimal Order Calculator</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        product = st.selectbox(
            "Product:",
            assistant.price_history['product'].unique(),
            key="order_product"
        )
        
        budget = st.number_input("Available Budget ($):", min_value=1000, max_value=1000000, value=50000)
        
        current_inventory = st.number_input("Current Inventory:", min_value=0, max_value=10000, value=100)
        
        if st.button("Calculate Optimal Order", type="primary"):
            optimal_order = assistant.calculate_optimal_order(product, budget, current_inventory)
            
            if optimal_order:
                st.success("🎯 AI Recommendation Generated!")
                
                st.metric("Optimal Order Quantity", f"{optimal_order['optimal_quantity']:.0f} units")
                st.metric("Best Order Month", f"Month {optimal_order['best_month']}")
                st.metric("Predicted Price", f"${optimal_order['predicted_price']:.2f}")
                st.metric("Total Cost", f"${optimal_order['total_cost']:.2f}")
                st.metric("Potential Savings", f"${optimal_order['savings_potential']:.2f}")

def show_supplier_analysis(assistant):
    st.markdown('<div class="sub-header">Supplier Performance Analysis</div>', unsafe_allow_html=True)
    
    # Supplier filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_reliability = st.slider("Minimum Reliability Score:", 1.0, 5.0, 3.0)
    
    with col2:
        max_delivery = st.slider("Maximum Delivery Days:", 1, 30, 15)
    
    with col3:
        risk_filter = st.multiselect(
            "Risk Level:",
            ['Low', 'Medium', 'High'],
            default=['Low', 'Medium']
        )
    
    # Filter suppliers
    filtered_suppliers = assistant.supplier_data[
        (assistant.supplier_data['reliability_score'] >= min_reliability) &
        (assistant.supplier_data['delivery_time_days'] <= max_delivery) &
        (assistant.supplier_data['risk_level'].isin(risk_filter))
    ]
    
    st.subheader(f"Filtered Suppliers ({len(filtered_suppliers)})")
    
    # Display supplier table
    st.dataframe(
        filtered_suppliers[['id', 'name', 'category', 'reliability_score', 'delivery_time_days', 'risk_level']],
        use_container_width=True
    )
    
    # Supplier performance chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(filtered_suppliers, x='reliability_score', y='delivery_time_days',
                        color='risk_level', size='cost_score',
                        title="Supplier Performance Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        category_performance = filtered_suppliers.groupby('category').agg({
            'reliability_score': 'mean',
            'delivery_time_days': 'mean'
        }).reset_index()
        
        fig = px.bar(category_performance, x='category', y='reliability_score',
                    title="Average Reliability by Category")
        st.plotly_chart(fig, use_container_width=True)

def show_demand_forecasting(assistant):
    st.markdown('<div class="sub-header">Demand Pattern Forecasting</div>', unsafe_allow_html=True)
    
    selected_product = st.selectbox(
        "Select Product for Demand Analysis:",
        assistant.demand_patterns['product'].unique(),
        key="demand_product"
    )
    
    product_demand = assistant.demand_patterns[assistant.demand_patterns['product'] == selected_product]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(product_demand, x='month', y='demand',
                     title=f"Seasonal Demand Pattern - {selected_product}",
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        accuracy = product_demand['forecast_accuracy'].mean()
        st.metric("Average Forecast Accuracy", f"{accuracy*100:.1f}%")
        
        avg_demand = product_demand['demand'].mean()
        st.metric("Average Monthly Demand", f"{avg_demand:.0f} units")
        
        peak_month = product_demand.loc[product_demand['demand'].idxmax(), 'month']
        st.metric("Peak Demand Month", peak_month)

if __name__ == "__main__":
    main()