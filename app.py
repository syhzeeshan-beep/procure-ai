import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="ProcureAI - Commercial Procurement Assistant",
    page_icon="📊",
    layout="wide"
)

# Simple procurement app without complex dependencies
def main():
    st.title("🚀 ProcureAI - Commercial Procurement Assistant")
    st.subheader("AI-Powered Procurement Analytics")
    
    # Sidebar
    st.sidebar.title("Navigation")
    menu = st.sidebar.selectbox("Choose Module", 
                               ["Dashboard", "Price Analysis", "Supplier Management"])
    
    if menu == "Dashboard":
        show_dashboard()
    elif menu == "Price Analysis":
        show_price_analysis()
    else:
        show_supplier_management()

def show_dashboard():
    st.header("Procurement Dashboard")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Suppliers", "156", "12 new")
    with col2:
        st.metric("Avg Delivery Time", "4.2 days", "-0.5 days")
    with col3:
        st.metric("Cost Savings", "15.2%", "2.1%")
    
    # Simple data table
    st.subheader("Recent Procurement Activity")
    data = {
        'Product': ['Laptops', 'Office Chairs', 'Monitors', 'Desks'],
        'Quantity': [50, 25, 30, 15],
        'Unit Price': [1200, 250, 300, 450],
        'Supplier': ['TechCorp', 'OfficePro', 'DisplayCo', 'FurnitureInc']
    }
    df = pd.DataFrame(data)
    st.dataframe(df)
    
    # Price trend simulation
    st.subheader("Price Trends")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    prices = [100, 105, 102, 98, 95, 92]
    
    st.line_chart(pd.DataFrame({'Price': prices}, index=months))

def show_price_analysis():
    st.header("Price Analysis & Predictions")
    
    product = st.selectbox("Select Product", 
                          ["Laptops", "Monitors", "Office Furniture", "Raw Materials"])
    
    st.subheader(f"Price Analysis for {product}")
    
    # Simple price simulation
    base_price = st.slider("Base Price", 100, 1000, 500)
    trend = st.selectbox("Predicted Trend", ["Decreasing", "Stable", "Increasing"])
    
    if trend == "Decreasing":
        prediction = "Prices expected to drop 5-10% in next quarter"
        color = "green"
    elif trend == "Increasing":
        prediction = "Prices expected to rise 5-10% in next quarter" 
        color = "red"
    else:
        prediction = "Prices expected to remain stable"
        color = "blue"
    
    st.info(f"**AI Prediction:** :{color}[{prediction}]")
    
    # Optimal order calculation
    st.subheader("Optimal Order Calculator")
    budget = st.number_input("Budget", min_value=1000, value=50000)
    current_inventory = st.number_input("Current Inventory", min_value=0, value=100)
    
    if st.button("Calculate Optimal Order"):
        optimal_qty = min(budget / base_price, current_inventory * 2)
        total_cost = optimal_qty * base_price * 0.95  # 5% bulk discount
        
        st.success(f"**Recommended Order:** {optimal_qty:.0f} units")
        st.success(f"**Estimated Cost:** ${total_cost:,.2f}")
        st.success(f"**Potential Savings:** ${optimal_qty * base_price * 0.05:,.2f}")

def show_supplier_management():
    st.header("Supplier Management")
    
    # Supplier data
    suppliers = {
        'Name': ['TechCorp', 'OfficePro', 'DisplayCo', 'FurnitureInc', 'SupplyMaster'],
        'Rating': [4.5, 4.2, 4.7, 4.0, 4.3],
        'Delivery Time': [3, 5, 2, 7, 4],
        'Risk Level': ['Low', 'Medium', 'Low', 'High', 'Low']
    }
    
    df = pd.DataFrame(suppliers)
    st.dataframe(df)
    
    # Supplier filters
    st.subheader("Supplier Analysis")
    min_rating = st.slider("Minimum Rating", 3.0, 5.0, 4.0)
    max_delivery = st.slider("Maximum Delivery Days", 1, 10, 5)
    
    filtered_suppliers = df[
        (df['Rating'] >= min_rating) & 
        (df['Delivery Time'] <= max_delivery)
    ]
    
    st.write(f"Found {len(filtered_suppliers)} matching suppliers")
    st.dataframe(filtered_suppliers)

if __name__ == "__main__":
    main()
