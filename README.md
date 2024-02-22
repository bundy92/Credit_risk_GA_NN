# Credit Risk Analysis Streamlit App

Welcome to the Credit Risk Analysis Streamlit App! This web application is designed to facilitate comprehensive credit risk analysis utilizing machine learning techniques. It provides a user-friendly interface for data uploading, model training, hyperparameter optimization, and evaluation of trained models.

## Features

### 1. Upload CSV File

- Users can seamlessly upload their datasets in CSV format for analysis.

### 2. Hyperparameter Optimization

- Employing a genetic algorithm, users can optimize hyperparameters crucial for machine learning models. They have the flexibility to adjust parameters such as population size, number of generations, mutation rate, and number of epochs.

### 3. Model Training

- Train logistic regression models using the uploaded dataset. The application intelligently selects hyperparameters based on optimization or employs default values if not optimized.

### 4. Model Evaluation

- Evaluate the trained models using a variety of performance metrics including accuracy, precision, recall, F1-score, AUC-ROC (Area Under the Receiver Operating Characteristic curve), and ROC (Receiver Operating Characteristic) curve.

### 5. Feature Importance

- Display feature importance charts based on the coefficients of logistic regression models, providing insights into the significance of different features in predicting credit risk.

## Installation

To run the application locally, follow these simple steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your_username/Credit_risk_GA_NN.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd Credit_risk_GA_NN
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app:**

    ```bash
    streamlit run streamlit_app.py
    ```

## Usage

1. **Upload Dataset:**
    - Click on the "Upload CSV file" button and select a CSV file containing your dataset for analysis.

2. **Optimize Hyperparameters (Optional):**
    - Users can optimize hyperparameters for the machine learning model using a genetic algorithm. Adjust hyperparameters in the sidebar, such as population size, number of generations, mutation rate, and number of epochs. Click on the "Optimize Hyperparameters" button to start the optimization process.

3. **Train Model:**
    - Click on the "Train Model" button to initiate training of a logistic regression model using the uploaded dataset. If hyperparameters are optimized, the optimized values will be utilized; otherwise, default values will be employed.

4. **View Model Evaluation Metrics:**
    - After model training, various evaluation metrics including accuracy, precision, recall, F1-score, and AUC-ROC (Area Under the Receiver Operating Characteristic curve) will be displayed in the sidebar. Additionally, the ROC curve, SHAP, PDPs and feature importance (if available) will be visualized.

## Contributors

- **Thomas Bundy** (@bundy92) - Project Lead

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by [OpenAI](https://openai.com/)
- Special thanks to the Streamlit and PyTorch team for their invaluable solutions!

## Support

For any inquiries or support, please don't hesitate to contact us.
