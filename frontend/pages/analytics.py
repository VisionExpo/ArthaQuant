import streamlit as st
from frontend.app import api_get

st.header("📈 Analytics")

data = api_get("/analytics")

st.metric("Sharpe Ratio", data["sharpe_ratio"])
st.metric("Max Drawdown", data["max_drawdown"])
st.metric("Total Return", data["total_return"])

st.line_chart(data["equity_curve"])
