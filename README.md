Linear and Polynomial Regression Visualisation
This project demonstrates how to visualise the predictions of two pre-trained regression models (linear and polynomial) using test data. Both models were trained on the same dataset and are compared on a plot to show how well each fits the actual values.

Project structure
visualise_regression/
├── data/                     # CSV files with test inputs and target values
├── models/                   # Pre-trained regression models (Pickle format)
├── output/                   # Final plot saved as a JPG image
├── visualise_regression.py   # Main script
└── README.md

The polynomial model uses transformed input features (e.g., x², x³), but the visualisation uses the original X values for both models to make the comparison meaningful.

Input values are sorted before plotting to ensure smooth regression lines.










