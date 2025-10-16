import streamlit as st
import pandas as pd
import pyarrow.parquet as pq

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

@st.cache_data
def get_data(filename):
    taxi_data = pq.read_table(filename)
    taxi_data = taxi_data.to_pandas()

    return taxi_data



with header:
    st.title("Welcome to Ronald 1st data science project on Streamlit")
    st.markdown("On computer's large screen? The **Wide mode** is good. Choose from the Settings, the 3 dots on top right.")

    

with dataset:
    st.header("NYC taxi dataset")
    st.text("I found this dataset on ...")

    # taxi_data = pd.read_csv('data/taxi_data.csv')
    # taxi_data = pq.read_table("data/yellow_tripdata_2020-11.parquet")
    # taxi_data = taxi_data.to_pandas()
    taxi_data = get_data("data/yellow_tripdata_2020-11.parquet")
    st.write(taxi_data.head(5))
    
    row_count = len(taxi_data)
    st.text(f"Total row count for dataset yellow_tripdata_2020-11 is : {row_count:,}")

    # --- Sampling function --- from ChatGPT
def sample_data(df, sample_frac=0.001):
    """Return a small sample of the full dataset (default 0.1%)."""
    #return df.sample(frac=sample_frac, random_state=42) ChatGPT problem: fixed 42, then always same sample
    return df.sample(frac=sample_frac) #Claude solved it, by removing it


# User controls
st.sidebar.header("Controls")
use_sample = st.sidebar.checkbox("Use sample data (0.1%) for testing", value=True)

# If user wants full dataset
if use_sample:
    taxi_data = sample_data(taxi_data)
    st.info(f"Running on sample: {len(taxi_data):,} rows ({100*len(taxi_data)/row_count:.2f}% of full dataset)")

else:
    taxi_data = taxi_data
    st.success(f"Running on full dataset: {len(taxi_data):,} rows")


    # Claude helped added a button for user to download the sample as csv
st.download_button(
    label="Download sample data as CSV",
    data=taxi_data.to_csv(index=False),
    file_name="sample_data.csv",
    mime="text/csv"
)

st.subheader("Pick-up location ID distribution on the NYC dataset")

pulocation_dist = pd.DataFrame(taxi_data["PULocationID"].value_counts())
st.bar_chart(pulocation_dist)

st.subheader("Pivot Table as first attempt to ask: ")

st.markdown(
        "* **which hours** good for changing shift ?"
    )

st.image("images/pivotTableRH.png")

with features:
    st.header("The features I created")

    st.markdown(
        "* **first feature:** I created this feature because of this .... I calculated using..."
    )
    st.markdown(
        "* **second feature:** I created this feature because of this .... I calculated using..."
    )

    url = "https://docs.google.com/document/d/1XkQLTE_Ug10nQst58iF4J_8TQ-Ee_9Puhga1qwwcjes/edit?tab=t.0"
    #st.write("check out this [link](%s)" % url)
    st.markdown("* **check out this:** for Ronald's research on Random Forest and bagging: [link](%s)" % url)


with model_training:
    st.header("Time to train the model")
    st.text(
        "Here you get to choose the hyperparameters of the model and see how the performance changes"
    )

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider(
        "What should be the max_depth of the model?",
        min_value=10,
        max_value=100,
        value=20,
        step=10,
    )

    n_estimators = sel_col.selectbox(
        "How many trees should there be?", options=[100, 200, 300, "No Limit"], index=0
    )
    sel_col.text('Here is a list of features in my data:')
    sel_col.write(taxi_data.columns)

    input_feature = sel_col.text_input(
        "Which feature should be used as the input feature?", "PULocationID"
    )

    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    X = taxi_data[[input_feature]]
    y = taxi_data[["trip_distance"]]

regr.fit(X, y)
prediction = regr.predict(X) #was y wrongly used in Misra tutorial, should be X
# ref. offical doc at
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.predict

disp_col.subheader("Mean absolute error of the model is:")
disp_col.write(mean_absolute_error(y, prediction))

disp_col.subheader("Mean squared error of the model is:")
disp_col.write(mean_squared_error (y, prediction))

disp_col.subheader("R squared score of the model is:")
disp_col.write(r2_score(y, prediction))
