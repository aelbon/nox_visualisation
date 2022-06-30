import pandas as pd
import pandas_profiling
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import io
import plotly.express as px
import numpy as np

from streamlit_pandas_profiling import st_profile_report

dir = "D://Projects Machine Learning//Forecasting//NN forecasting//Trials//"
df = pd.read_csv(dir + "subset.csv")
df['date_time'] = pd.to_datetime(df['date_time'])
columns = ["NO2", "CO", "O3", "PM10", "PM2.5"]

st.title("NOX DATA")

# Display the dataframe
sidebar = st.sidebar
df_display = sidebar.checkbox("Display Raw Data", value=True)
df_display_Info = sidebar.checkbox("Display Data Description", value=True)
df_display_Prof = sidebar.checkbox("Pandas Profiling Report ", value=False)
df_display_Pairplot = sidebar.checkbox("Seaborn Pair Plot", value=False)
df_display_Lineplots = sidebar.checkbox("Seaborn Line Plots", value=False)
df_display_MovingAverages = sidebar.checkbox("Plotly moving averages", value=False)
df_display_missing = sidebar.checkbox("Missing plot", value=False)
df_display_Boxplots = sidebar.checkbox("Box plots", value=False)
df_display_Anomalies = sidebar.checkbox("Anomaly detection with Sliding Windows", value=False)

@st.experimental_singleton
def get_pandas_report():
    return df[columns].profile_report(config_file="pandas_profiling_config.yml")


@st.experimental_memo
def get_pair_plot(var_columns):
    return sns.pairplot(df, vars=var_columns)


if df_display:
    st.header("Dataframe")
    st.write(df)

if df_display_Info:
    st.header("Data Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("Data Description")
    st.write(df.describe())

    st.subheader("Data Missing Values Count")
    st.table(df.isna().sum())

if df_display_Prof:
    st.write("Report")
    # pr = df[columns].profile_report(config_file="pandas_profiling_config.yml")
    pr = get_pandas_report()
    st_profile_report(pr)

if df_display_Pairplot:
    # NOTICE the way fig is used here!
    # default
    st.write("Pair plot")
    fig = get_pair_plot(columns)
    st.pyplot(fig)

# changed the layout a bit (it is not easy to get what you want)
# fig = sns.pairplot(df[columns], height=1.8, aspect=1.5)
# sns.set(font_scale=1.4)
# st.pyplot(fig)

# distribution example for just one element
# fig = sns.displot(x="NO2", data=df, height=2.5, aspect=1.75)
# st.pyplot(fig)

if df_display_Lineplots:
    st.write("Line plots")
    fig = plt.figure(figsize=(6, 4))
    sns.lineplot(x="date_time", y="NO2", data=df)
    st.pyplot(fig)

    fig = plt.figure(figsize=(6, 4))
    sns.lineplot(x="date_time", y="CO", data=df)
    st.pyplot(fig)

# alternativ:
# st.line_chart(df['CO'])

if df_display_MovingAverages:
    st.write("MovingAverages")
    # create
    df['MA48'] = df['NO2'].rolling(48).mean()
    df['MA336'] = df['NO2'].rolling(336).mean()

    fig = px.line(df, x="date_time", y=['NO2', 'MA48', 'MA336'], title='NO2', template='plotly_dark')
    st.plotly_chart(fig)

df.set_index('date_time', inplace=True)

if df_display_missing:
    ### PLOT MISSING VALUES OVER TIME ###
    st.write("Missing values over time")
    fig = plt.figure(figsize=(18, 5))
    sns.heatmap(df[['NO2']].isna().T, cbar=False, cmap='inferno',
                xticklabels=False, yticklabels=['NO2'])
    plt.xticks(range(0, len(df), 24 * 30), list(df.index.date[::24 * 30]))
    # np.set_printoptions(False)
    st.pyplot(fig)

if df_display_Boxplots:
    st.write("Box plots")
    fig = plt.figure(figsize=(9, 5))
    sns.boxplot(x=df.index.hour, y=df.NO2, palette='autumn')
    plt.ylabel('NO2')
    plt.xlabel('hour')
    st.write(fig)

    fig = plt.figure(figsize=(9, 5))
    sns.boxplot(x=df.index.hour, y=df.CO, palette='autumn')
    plt.ylabel('CO')
    plt.xlabel('hour')
    st.write(fig)

if df_display_Anomalies:
    column_data = df['NO2'].to_numpy()
    time = np.arange(0, len(column_data))

    def get_bands(data):
        mean = np.nanmean(data)
        std = 3*np.nanstd(data)
        return mean + std, mean - std

    def get_bands_alt(data):
        mean = np.nanmean(data)
        quantile = np.nanquantile(data, 0.98)
        return mean + quantile, mean - quantile


    @st.experimental_memo
    def get_upper_lower(window_percentage, data):
        N = len(data)
        k = int(N * (window_percentage/100))
        bands = [get_bands(data[range(0 if i - k < 0 else i - k, i + k if i + k < N else N)]) for i in range(0, N)]
        return zip(*bands)

    # compute local outliers
    upper, lower = get_upper_lower(10, column_data)
    anomalies = (column_data > upper) | (column_data < lower)

    # plotting...
    fig = plt.figure(figsize=(20,10))
    plt.plot(time, column_data, 'k', label='Data')
    plt.plot(time, upper, 'r-', label='Bands', alpha=0.3)
    plt.plot(time, lower, 'r-', alpha=0.3)

    plt.plot(time[anomalies], column_data[anomalies], 'ro', label='Anomalies')
    plt.fill_between(time, upper, lower, facecolor='red', alpha=0.1)
    plt.legend()

    st.pyplot(fig)