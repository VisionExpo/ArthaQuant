import streamlit as st
from datetime import datetime
from frontend.app import api_post

st.header("💼 Paper Trade")

symbol = st.text_input("Symbol", "AAPL")
price = st.number_input("Current Price", value=100.0)

p_up = st.slider("P(Up)", 0.0, 1.0, 0.75)
expected_return = st.number_input("Expected Return", value=0.02)
uncertainty = st.number_input("Uncertainty", value=0.1)

if st.button("Execute Paper Trade"):
    payload = {
        "symbol": symbol,
        "p_up": p_up,
        "expected_return": expected_return,
        "uncertainty": uncertainty,
        "price": price,
        "timestamp": datetime.utcnow().isoformat(),
    }

    out = api_post("/paper/trade", payload)
    st.json(out)
