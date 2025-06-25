import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Sample data setup
baskets = {
    "Portfolio A": {
        "values": [1.1, 2.3, 3.3],
        "timeseries": pd.DataFrame({
            "Date": pd.date_range(start="2023-01-01", periods=10),
            "Value": np.random.rand(10)*100
        }),
        "components": {
            "2023-01-01": ["Apple", "Banana"],
            "2023-01-02": ["Orange", "Grapes"],
        }
    },
    "Portfolio B": {
        "values": [4.4, 5.2],
        "timeseries": pd.DataFrame({
            "Date": pd.date_range(start="2023-01-01", periods=10),
            "Value": np.random.rand(10)*50
        }),
        "components": {
            "2023-01-01": ["Bread", "Butter"],
            "2023-01-02": ["Cheese", "Ham"],
        }
    }
}

# Sidebar selection
selected_basket = st.sidebar.selectbox("Select Portfolio", list(baskets.keys()))
basket_data = baskets[selected_basket]

# Display numeric values
st.write(f"**Historic Values for {selected_basket}:**")
st.table(pd.DataFrame({"Values": basket_data["values"]}))

# Display time series chart
ts = basket_data["timeseries"]
chart = alt.Chart(ts).mark_line().encode(
    x="Date:T",
    y="Value:Q",
    tooltip=["Date:T", "Value:Q"]
).interactive()

st.altair_chart(chart, use_container_width=True)

# Date click selection simulation
selected_date = st.date_input(
    "Select a date to view components",
    value=ts["Date"].iloc[0],
    min_value=ts["Date"].min(),
    max_value=ts["Date"].max()
).strftime('%Y-%m-%d')
components = basket_data["components"].get(selected_date, ["No data available for this date."])
st.write(f"**Components for {selected_date}:**")
st.table(pd.DataFrame({"Components": components}))