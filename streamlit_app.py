# streamlit_app.py
import sys

import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from scipy.cluster import hierarchy
import shap
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
import seaborn as sns
from hyper_v2 import load_data, train_pd_model, evaluate_pd_model, preprocess_data, genetic_algorithm_hyperparameter_optimization

st.set_option('deprecation.showPyplotGlobalUse', False)

# Main function
def main():
    st.title("Credit Risk Analysis")

    # Set flag.
    hyperparams_optimized = False
    best_model_params, best_auc_roc = None, None

    # File upload and data loading
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Dataset Preview:", data.head())

        # Sidebar controls for optimization
        button_optimize = st.sidebar.button("Optimize Hyperparameters")
        population_size = st.sidebar.slider("Population Size", min_value=10, max_value=1000, value=100, step=10)
        num_generations = st.sidebar.slider("Number of Generations", min_value=1, max_value=100, value=50, step=1)
        mutation_rate = st.sidebar.slider("Mutation Rate", min_value=0.01, max_value=0.1, value=0.01, step=0.01)
        num_epochs = st.sidebar.slider("Number of Epochs", min_value=10, max_value=1000, value=10, step=10)

        if button_optimize:
            st.subheader("Hyperparameter Optimization")
            st.write("This process may take some time. (1 generation ~ 15 seconds) Please wait...")

            # Preprocess data
            X, y = preprocess_data(data)
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

            # Perform hyperparameter optimization
            best_model_params, best_auc_roc = genetic_algorithm_hyperparameter_optimization(X_train, y_train, X_val, y_val,
                                                                                        input_size=X_train.shape[1],
                                                                                        hidden_sizes=[256, 128, 64, 32],
                                                                                        output_size=1,
                                                                                        population_size=population_size,
                                                                                        num_generations=num_generations,
                                                                                        mutation_rate=mutation_rate,
                                                                                        learning_rates=[0.001, 0.01, 0.1],
                                                                                        num_epochs=num_epochs)

            # Check if hyperparameter optimization was successful
            if best_model_params:
                # If optimization was successful, set flag to indicate hyperparameters are optimized
                hyperparams_optimized = True
                st.write(f"Best AUC-ROC on Validation Set: {best_auc_roc:.4f}")
                #st.write(f"Best Model Parameters: {best_model_params}")
                st.write("Optimization completed!")
            else:
                # If optimization was not successful, inform the user
                st.warning("Hyperparameter optimization failed. Please try again.")

            
        # Sidebar controls for training
        button_train = st.sidebar.button("Train Model")
        if button_train:
            st.subheader("Model Training")

            if hyperparams_optimized:
                st.warning("Please optimize hyperparameters first.")
            else:
                X, y = preprocess_data(data)
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

                # # Cluster Analysis Dendrogram
                st.write("Cluster analysis dendrogram demonstrating the hierarchical structure of borrower segments based on their credit risk profiles.")
                st.warning("Cluster analysis removed as it takes too long to compute.")  
                # linkage_matrix = hierarchy.linkage(X_train, method='ward')  # Compute the linkage matrix
                # plt.figure(figsize=(10, 6))
                # plt.title('Hierarchical Clustering Dendrogram')
                # plt.xlabel('Sample Index')
                # plt.ylabel('Distance')
                # hierarchy.dendrogram(linkage_matrix, ax=plt.gca())
                # st.pyplot(bbox_inches='tight', pad_inches=0)


                # Sankey Diagram
                st.write("Sankey diagram visualizing the flow of borrowers across different risk categories or credit scoring tiers.")
                st.warning("Sankey diagram implementation code goes here.")  # Add your Sankey diagram code

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

        # Sidebar controls for displaying the Sankey diagram
                    
        # Assuming you have your data in a format suitable for constructing the Sankey diagram
        # For example, a list of tuples where each tuple represents a flow (source, target, value)

        # # Example data
        # flows = [
        #     ('Low Risk', 'Approved', 500),
        #     ('Low Risk', 'Rejected', 50),
        #     ('Medium Risk', 'Approved', 300),
        #     ('Medium Risk', 'Rejected', 100),
        #     ('High Risk', 'Approved', 50),
        #     ('High Risk', 'Rejected', 200),
        # ]
        # display_sankey = st.sidebar.checkbox("Display Sankey Diagram")

        # if display_sankey:
        #     st.subheader("Sankey Diagram")

        #     # Create Sankey diagram
        #     sankey = Sankey(flows=flows, scale=0.01)  # Scale adjusts the width of the flow paths
        #     fig, ax = plt.subplots()
        #     sankey.finish()
            
        #     # Show the diagram
        #     st.pyplot(fig)

if __name__ == "__main__":
    main()