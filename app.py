import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import pickle
import shap

# Load the trained model
model = pickle.load(open('./model/final_model.pkl', 'rb'))

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Risk Predictor")
st.write("Predict churn risk for an Iranian telecom customer and estimate retention cost.")

tab1, tab2 = st.tabs(["Single Customer", "Batch Upload"])

with tab1:

    # ----- input fields for user to enter customer data -----
    st.header("Customer Information")
    with st.form(key='customer_form'):
        col1, col2 = st.columns(2)
        with col1:
            call_failure = st.number_input("Call Failure", min_value=0, value=50)
            subscription_length = st.number_input("Subscription Length (months)", min_value=0, value=50)
            status = st.selectbox("Status", [1, 2])
            customer_value = st.number_input("Customer Value", min_value=0.0, value=3000.0)
            seconds_of_use = st.number_input("Seconds of Use", min_value=0, value=20000)
            frequency_of_use = st.number_input("Frequency of Use", min_value=0, value=300)
            frequency_of_sms = st.number_input("Frequency of SMS", min_value=0, value=600)
        
        with col2:
            complains = st.selectbox("Complains", [0, 1])
            distinct_called = st.number_input("Distinct Called Numbers", min_value=0, value=15)
            age_group = st.selectbox("Age Group", [1, 2, 3, 4, 5])
            tariff_plan = st.selectbox("Tariff Plan", [1, 2])
            charge_amount = st.slider("Charge Amount (band)", 0, 10, 3)
            age = st.number_input("Age", min_value=10, max_value=100, value=30)
            
        submitted = st.form_submit_button("Predict Churn Risk")

        # prediction logic
        if submitted:
            # Create a DataFrame from the input values
            input_data = pd.DataFrame({
                'Call Failure': [call_failure],
                'Complains': [complains],
                'Subscription Length': [subscription_length],
                'Charge Amount': [charge_amount],
                'Seconds of Use': [seconds_of_use],
                'Frequency of use': [frequency_of_use],
                'Frequency of SMS': [frequency_of_sms],
                'Distinct Called Numbers': [distinct_called],
                'Age Group': [age_group],
                'Tariff Plan': [tariff_plan],
                'Status': [status],
                'Age': [age],
                'Customer Value': [customer_value]
            })

            # Make prediction
            prob = model.predict_proba(input_data)[0][1]
            threshold = 0.5  # adjust based on your chosen operating point

            st.subheader("Prediction Result")
            st.metric("Churn Probability", f"{prob*100:.1f}%")

            if prob >= threshold:
                st.error("⚠️ High churn risk — recommend retention outreach")
                offer_cost = customer_value * 0.20
                st.write(f"Estimated retention offer cost: **${offer_cost:.2f}**")
            else:
                st.success("✅ Low churn risk")
            
            # Why this customer is predicted to churn
            st.subheader("Why this Prediction")

            explainer = pickle.load(open("./model/shap_explainer.pkl", "rb"))
            
            explaination = explainer(input_data)
            fig, ax = plt.subplots()

            if explaination.values.ndim == 3:
                shap.plots.waterfall(explaination[0,:,1], max_display = 8, show = False)

            else:
                shap.plots.waterfall(explaination[0], max_display = 8, show = False)


            st.pyplot(fig)

            top_features = pd.Series(explaination.values[0, :, 1], index=input_data.columns).sort_values(ascending=False)
            top_driver = top_features.index[0]
            direction = "increased" if top_features.iloc[0] > 0 else "decreased"
            st.write(f"**Main driver:** `{top_driver}` {direction} this customer's churn risk the most.")

    pass

with tab2:

    st.subheader("Upload Customer Data")
    st.write("Upload a CSV with the same columns as the training data to get churn predictions for all customers, ranked by risk and value.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:

        batch_df = pd.read_csv(uploaded_file)

        required_cols = ['Call Failure', 'Complains', 'Subscription Length', 'Charge Amount',
                          'Seconds of Use', 'Frequency of use', 'Frequency of SMS',
                          'Distinct Called Numbers', 'Age Group', 'Tariff Plan', 'Status',
                          'Age', 'Customer Value']

        batch_df.columns = batch_df.columns.str.replace("  ", " ")
        missing = [c for c in required_cols if c not in batch_df.columns]

        if missing:
            st.error(f"""Missing required columns: {missing}
            This Csv columns are: {batch_df.columns}
            """)

        else:

            batch_df["Churn Probability"] = model.predict_proba(batch_df[required_cols])[:,1]

            threshold_slider = st.slider("Churn Risk Probability",0.0,1.0,0.5,0.05)

            batch_df["At Risk"] = batch_df["Churn Probability"] >= threshold_slider

            # Business framing: cost + priority
            offer_cost_pct = 0.20
            batch_df['Est. Offer Cost'] = batch_df['Customer Value'] * offer_cost_pct

            at_risk_df = batch_df[batch_df['At Risk']].sort_values('Customer Value', ascending=False)

            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Customers", len(batch_df))
            col2.metric("Flagged At-Risk", len(at_risk_df))
            col3.metric("Est. Total Retention Cost", f"${at_risk_df['Est. Offer Cost'].sum():,.0f}")

            budget = 15000
            if at_risk_df['Est. Offer Cost'].sum() > budget:
                affordable = at_risk_df[at_risk_df['Est. Offer Cost'].cumsum() <= budget]
                st.warning(f"⚠️ Flagged customers exceed the ${budget:,} budget. "
                           f"Showing top {len(affordable)} highest-value at-risk customers you can afford to target.")
                display_df = affordable
            else:
                st.success(f"✅ Within budget — ${budget - at_risk_df['Est. Offer Cost'].sum():,.0f} remaining.")
                display_df = at_risk_df

            st.subheader("At-Risk Customers (Ranked by Value)")
            st.dataframe(
                display_df[['Customer Value', 'Churn Probability', 'Est. Offer Cost']].style.format({
                    'Customer Value': '{:.0f}',
                    'Churn Probability': '{:.1%}',
                    'Est. Offer Cost': '${:.2f}'
                }),
                use_container_width=True
            )

            # Download button
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Prioritized List", csv, "at_risk_customers.csv", "text/csv")
