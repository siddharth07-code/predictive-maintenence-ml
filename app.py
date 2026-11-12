import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)


logistic_model = joblib.load("logistic_regression.pkl")
decision_tree = joblib.load("decision_tree.pkl")
random_forest = joblib.load("random_forest.pkl")
xgb_model = joblib.load("xgboost.pkl")
scaler = joblib.load("scaler.pkl")


st.sidebar.title("⚙️ Predictive Maintenance")

st.sidebar.write(
    "Machine failure prediction using Machine Learning."
)

model_name = st.sidebar.selectbox(
    "Choose ML Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ]
)


st.title("⚙️ Predictive Maintenance System")

st.write(
    "Enter the operating parameters of a machine "
    "to predict the probability of machine failure."
)

st.divider()

EXAMPLES = {
    "Normal operation (ID 1)": {
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 0,
        "machine_type": "M"
    },
    "Recorded failure (ID 51)": {
        "air_temperature": 298.9,
        "process_temperature": 309.1,
        "rotational_speed": 2861,
        "torque": 4.6,
        "tool_wear": 143,
        "machine_type": "L"
    }
}


def load_example():
    example = st.session_state.example
    if example is not None:
        st.session_state.update(EXAMPLES[example])
        st.session_state.example = None


for key, default in {
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1500,
    "torque": 40.0,
    "tool_wear": 100,
    "machine_type": "H"
}.items():
    st.session_state.setdefault(key, default)


with st.columns(2)[0]:
    st.selectbox(
        "Load example",
        list(EXAMPLES),
        index=None,
        placeholder="Choose an example",
        key="example",
        on_change=load_example
    )

st.subheader("Machine Parameters")

col1, col2 = st.columns(2)


with col1:

    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        key="air_temperature"
    )

    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        key="process_temperature"
    )

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=0,
        max_value=5000,
        key="rotational_speed"
    )


with col2:

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        key="torque"
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=300,
        key="tool_wear"
    )

    machine_type = st.selectbox(
        "Machine Type",
        ["H", "L", "M"],
        key="machine_type"
    )


st.divider()


if st.button(
    "🔍 Predict Machine Failure",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
        "Type_H": [int(machine_type == "H")],
        "Type_L": [int(machine_type == "L")],
        "Type_M": [int(machine_type == "M")]
    })


    if model_name == "Logistic Regression":

        input_scaled = scaler.transform(input_data)

        prediction = logistic_model.predict(
            input_scaled
        )[0]

        probability = logistic_model.predict_proba(
            input_scaled
        )[0][1]


    elif model_name == "Decision Tree":

        prediction = decision_tree.predict(
            input_data
        )[0]

        probability = decision_tree.predict_proba(
            input_data
        )[0][1]


    elif model_name == "Random Forest":

        prediction = random_forest.predict(
            input_data
        )[0]

        probability = random_forest.predict_proba(
            input_data
        )[0][1]


    else:

        input_xgb = input_data.copy()

        input_xgb.columns = [
            "air_temperature",
            "process_temperature",
            "rotational_speed",
            "torque",
            "tool_wear",
            "type_H",
            "type_L",
            "type_M"
        ]

        prediction = xgb_model.predict(
            input_xgb
        )[0]

        probability = xgb_model.predict_proba(
            input_xgb
        )[0][1]


    st.divider()

    st.subheader(
        f"Prediction using {model_name}"
    )

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        if prediction == 1:

            st.error(
                "⚠️ MACHINE FAILURE PREDICTED"
            )

        else:

            st.success(
                "✅ MACHINE OPERATING NORMALLY"
            )


    with result_col2:

        st.metric(
            "Failure Probability",
            f"{probability * 100:.2f}%"
        )


    st.write("Failure Probability")

    st.progress(
        float(probability)
    )


    st.divider()

    st.subheader("Machine Information")

    summary_col1, summary_col2 = st.columns(2)


    with summary_col1:

        st.write(
            f"**Machine Type:** {machine_type}"
        )

        st.write(
            f"**Air Temperature:** {air_temperature} K"
        )

        st.write(
            f"**Process Temperature:** {process_temperature} K"
        )


    with summary_col2:

        st.write(
            f"**Rotational Speed:** {rotational_speed} rpm"
        )

        st.write(
            f"**Torque:** {torque} Nm"
        )

        st.write(
            f"**Tool Wear:** {tool_wear} min"
        )
