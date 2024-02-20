# Credit Risk Analysis Streamlit App

This Streamlit web application is designed for credit risk analysis using machine learning models. The app provides functionalities for data upload, model training, hyperparameter optimization, and evaluation of the trained model.

## Features

- **Upload CSV File**: Users can upload their dataset in CSV format for analysis.
- **Hyperparameter Optimization**: Users can optimize hyperparameters for the machine learning model using a genetic algorithm. They can adjust parameters such as population size, number of generations, mutation rate, and number of epochs.
- **Model Training**: Train a logistic regression model using the uploaded dataset. If hyperparameters are optimized, the optimized values will be used; otherwise, default values will be used.
- **Model Evaluation**: Evaluate the trained model using various metrics including accuracy, precision, recall, F1-score, AUC-ROC (Area Under the Receiver Operating Characteristic curve), and ROC (Receiver Operating Characteristic) curve.
- **Feature Importance**: If available, display the feature importance chart based on the coefficients of the logistic regression model.

## Installation

To run the application locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/Credit_risk_GA_NN.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Credit_risk_GA_NN
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:

    ```bash
    streamlit run streamlit_app.py
    ```

## Usage

1. **Upload Dataset**: Click on the "Upload CSV file" button and select a CSV file containing your dataset.

2. **Optimize Hyperparameters (Optional)**: Adjust hyperparameters in the sidebar (e.g., population size, number of generations, mutation rate, number of epochs), then click on the "Optimize Hyperparameters" button to start the optimization process.

3. **Train Model**: Click on the "Train Model" button to train a logistic regression model using the uploaded dataset. If hyperparameters are optimized, the optimized values will be used.

4. **View Model Evaluation Metrics**: After training the model, evaluation metrics such as accuracy, precision, recall, F1-score, and AUC-ROC will be displayed in the sidebar. Additionally, the ROC curve and feature importance (if available) will be shown.

## Contributors

- Thomas Bundy (@bundy92) - Project Lead

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by [OpenAI](https://openai.com/)
- Special thanks to the Streamlit and Pytorch team for their amazing work!

## Support

For any inquiries or support, please contact me.
