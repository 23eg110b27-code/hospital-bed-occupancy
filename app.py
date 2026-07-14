import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go

st.set_page_config(page_title="Hospital Bed Occupancy Forecasting",
                   page_icon="🏥",
                   layout="wide")

st.title("🏥 Hospital Bed Occupancy Forecasting Using LSTM")

uploaded_file = st.file_uploader(
    "Upload bed occupancy.csv",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    target = df["Occupancy"].values.reshape(-1,1)

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(target)

    sequence_length = 5

    X=[]
    y=[]

    for i in range(sequence_length,len(scaled)):
        X.append(scaled[i-sequence_length:i])
        y.append(scaled[i])

    X=np.array(X)
    y=np.array(y)

    split=int(len(X)*0.8)

    X_train=X[:split]
    X_test=X[split:]

    y_train=y[:split]
    y_test=y[split:]

    model=Sequential()

    model.add(
        LSTM(
            64,
            activation="tanh",
            input_shape=(sequence_length,1)
        )
    )

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    with st.spinner("Training LSTM Model..."):
        model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=8,
            verbose=0
        )

    pred=model.predict(X_test)

    pred=scaler.inverse_transform(pred)
    actual=scaler.inverse_transform(y_test)

    st.success("Model Trained Successfully!")

    fig=go.Figure()

    fig.add_trace(
        go.Scatter(
            y=actual.flatten(),
            mode="lines",
            name="Actual"
        )
    )

    fig.add_trace(
        go.Scatter(
            y=pred.flatten(),
            mode="lines",
            name="Predicted"
        )
    )

    fig.update_layout(
        title="Actual vs Predicted Bed Occupancy",
        xaxis_title="Time",
        yaxis_title="Occupancy"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Predict Next Day Occupancy")

    seq=scaled[-sequence_length:]
    seq=seq.reshape(1,sequence_length,1)

    next_pred=model.predict(seq)

    next_occ=scaler.inverse_transform(next_pred)[0][0]

    st.metric(
        "Predicted Next Day Bed Occupancy",
        f"{next_occ:.0f} Beds"
    )

    st.subheader("Dataset Statistics")
    st.write(df.describe())