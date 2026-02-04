import streamlit as st
import numpy as np
from frontend.app import api_post

st.header("🔮 Predict")

symbol = st.text_input("Symbol", "AAPL")

if st.button("Run Prediction"):
    market_seq = np.random.randn(60, 10).tolist()

    payload = {
        "symbol": symbol,
        "market_sequence": market_seq,
        "input_ids": [101] * 128,
        "attention_mask": [1] * 128,
    }

    out = api_post("/predict", payload)

    st.metric("P(Up)", f"{out['p_up']:.2f}")
    st.metric("Expected Return", f"{out['expected_return']:.4f}")
    st.metric("Uncertainty", f"{out['uncertainty']:.4f}")
