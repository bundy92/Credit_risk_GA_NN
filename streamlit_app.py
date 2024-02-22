# Importing dependencies
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
import shap
from hyper_v3 import load_data, train_pd_model, evaluate_pd_model, preprocess_data, genetic_algorithm_hyperparameter_optimization, plot_line_plot, plot_bar_chart, plot_heatmap

# Turning off the error message
st.set_option('deprecation.showPyplotGlobalUse', False)

# Main function
def main():
    """
    Main function for the Credit Risk Analysis application.

    This function allows users to upload a CSV file, preprocesses the data,
    performs hyperparameter optimization using a genetic algorithm, trains
    the model, and displays evaluation metrics and visualizations.

    Returns:
    None
    """
    # Set title for the web application
    st.title("Credit Risk Analysis")

    # Set flag to indicate whether hyperparameters are optimized
    hyperparams_optimized = False
    best_model_params, best_auc_roc = None, None

    # File upload and data loading
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        # Load data from the uploaded file
        data = load_data(uploaded_file)
        st.write("Dataset Preview:", data.head())

        # Sidebar controls for selecting target column
        target_column = st.sidebar.selectbox("Select Target Column", data.columns)

        # Sidebar controls for hyperparameter optimization
        population_size = st.sidebar.slider("Population Size", min_value=10, max_value=1000, value=100, step=10)
        num_generations = st.sidebar.slider("Number of Generations", min_value=1, max_value=100, value=50, step=1)
        mutation_rate = st.sidebar.slider("Mutation Rate", min_value=0.01, max_value=0.1, value=0.01, step=0.01)
        num_epochs = st.sidebar.slider("Number of Epochs", min_value=10, max_value=1000, value=10, step=10)

        tab1, tab2, tab3 = st.tabs(["Optimization", "Training", "Benchmark"])

        # Buttons
        button_optimize = st.sidebar.button("Optimize Hyperparameters")
        button_train = st.sidebar.button("Train Model")
        button_benchmark = st.sidebar.button("Run Benchmark")

        # Optimization tab content
        with tab1:

            if button_optimize:
                # Perform hyperparameter optimization
                st.subheader("Hyperparameter Optimization")
                st.write("This process may take some time. (1 generation ~ 15 seconds) Please wait...")

                # Preprocess data
                X, y = preprocess_data(data, target_column)
                X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
                hyperparameters = ['Learning Rate','Hidden Layer Size', 'Batch Size']  # Names of the hyperparameters

                # Perform hyperparameter optimization
                best_model_params, best_auc_roc, initial_hyperparameters, optimized_hyperparameters = genetic_algorithm_hyperparameter_optimization(X_train, y_train, X_val, y_val,
                                                                                                                                                input_size=X_train.shape[1],
                                                                                                                                                hidden_sizes=[128, 64, 32],
                                                                                                                                                output_size=1,
                                                                                                                                                population_size=population_size,
                                                                                                                                                num_generations=num_generations,
                                                                                                                                                mutation_rate=mutation_rate,
                                                                                                                                                learning_rates=[0.001, 0.01, 0.1],
                                                                                                                                                batch_sizes=[128, 64, 32],
                                                                                                                                                num_epochs=num_epochs)

                # Plot line plot
                plot_line_plot(initial_hyperparameters, optimized_hyperparameters, hyperparameters)

                # Plot bar chart
                plot_bar_chart(initial_hyperparameters, optimized_hyperparameters, hyperparameters)

                # Plot heatmap
                plot_heatmap(initial_hyperparameters, optimized_hyperparameters, hyperparameters)

                # Check if hyperparameter optimization was successful
                if best_model_params:
                    # If optimization was successful, set flag to indicate hyperparameters are optimized
                    hyperparams_optimized = True
                    st.write(f"Best AUC-ROC on Validation Set: {best_auc_roc:.4f}")
                    st.write("Optimization completed!")
                else:
                    # If optimization was not successful, inform the user
                    st.warning("Hyperparameter optimization failed. Please try again.")

        # Training tab content
        with tab2:

            if button_train:
                # Train the model
                st.subheader("Model Training")

                if hyperparams_optimized:
                    st.warning("Please optimize hyperparameters first.")
                else:
                    X, y = preprocess_data(data, target_column)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    pd_model = train_pd_model(X_train, y_train, best_model_params)
                    accuracy, precision, recall, f1, auc_roc = evaluate_pd_model(pd_model, X_test, y_test)

                    st.sidebar.write("Accuracy:", accuracy)
                    st.sidebar.write("Precision:", precision)
                    st.sidebar.write("Recall:", recall)
                    st.sidebar.write("F1-score:", f1)
                    st.sidebar.write("AUC-ROC:", auc_roc)

                    # Model Interpretability Visualizations
                    st.subheader("Model Interpretability")

                    # Partial Dependence Plots (PDPs)
                    st.write("Partial Dependence Plots (PDPs) illustrating the impact of individual features on the predicted probability of default.")
                    feature_names = X.columns
                    shap.initjs()
                    explainer = shap.Explainer(pd_model, X_train)
                    shap_values = explainer.shap_values(X_train)
                    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
                    st.pyplot(bbox_inches='tight', pad_inches=0)

                    # SHAP Summary Plot
                    st.write("SHAP (SHapley Additive exPlanations) summary plot providing a global view of feature importances and their effects on model predictions.")
                    shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type='bar', show=False)
                    st.pyplot(bbox_inches='tight', pad_inches=0)

                    # Risk Segmentation Visualizations
                    st.subheader("Risk Segmentation")

                    # Display an explanation for the user
                    st.subheader("Model Evaluation Metrics")
                    st.write("Below is a bar chart showing the model evaluation metrics including Accuracy, Precision, Recall, F1-score, and AUC-ROC (Area Under the Receiver Operating Characteristic curve). These metrics help assess the performance of the trained model on the test data.")

                    # Create a DataFrame with the metrics
                    metrics_df = pd.DataFrame({
                        "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"],
                        "Value": [accuracy, precision, recall, f1, auc_roc]
                    })

                    # Display the metrics as a bar chart
                    st.bar_chart(metrics_df.set_index("Metric"))

                    # Calculate the ROC curve
                    fpr, tpr, thresholds = roc_curve(y_test, pd_model.predict_proba(X_test)[:, 1])
                    roc_auc = auc(fpr, tpr)
                    # Display the ROC curve
                    st.subheader("ROC Curve")
                    st.write("The ROC curve illustrates the trade-off between true positive rate (sensitivity) and false positive rate (1 - specificity).")
                    st.write("The area under the ROC curve (AUC) quantifies the overall performance of the classifier.")
                    st.write(f"AUC: {roc_auc:.2f}")

                    # Plot ROC curve
                    roc_df = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})
                    st.line_chart(roc_df.set_index("False Positive Rate"))
                    st.line_chart(roc_df)

                    # Check if the model has coefficients (for feature importance)
                    if hasattr(pd_model, 'coef_'):
                        # If coefficients are available, display feature importance
                        feature_importance = pd.Series(pd_model.coef_[0], index=X.columns)
                        st.subheader("Feature Importance")
                        st.write("The feature importance chart below shows the relative importance of each feature in the trained model. Features with higher coefficients are considered more important in predicting the target variable.")
                        st.bar_chart(feature_importance)
                    else:
                        # If coefficients are not available, inform the user
                        st.write("Feature importance is not available for this model.")

        # Benchmark tab content
        with tab3:
                
            if button_benchmark:
                # Run Benchmark button
                st.subheader("Benchmark Comparison")

                # Preprocess data
                X, y = preprocess_data(data, target_column)
                X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
                hyperparameters = ['Learning Rate','Hidden Layer Size', 'Batch Size']  # Names of the hyperparameters

                # Train initial model (before optimization)
                initial_model = train_pd_model(X_train, y_train)

                # Evaluate initial model
                initial_metrics = evaluate_pd_model(initial_model, X_val, y_val)

                # Evaluate optimized model
                optimized_model = train_pd_model(X_train, y_train, best_model_params)
                optimized_metrics = evaluate_pd_model(optimized_model, X_val, y_val)

                # Display benchmark comparison results
                st.write("Initial Model Metrics:")
                st.write(initial_metrics)
                st.write("Optimized Model Metrics:")
                st.write(optimized_metrics)

if __name__ == "__main__":
    main()
