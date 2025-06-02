# Linear and Polynomial Regression Visualisation

This project demonstrates how to visualise the predictions of two pre-trained regression models (linear and polynomial) using test data. Both models were trained on the same dataset and are compared on a plot to show how well each fits the actual values.

## Project structure

```
visualise_regression/
├── data/                      # Test CSV files
│   ├── linear_test_x.csv
│   ├── polynomial_test_x.csv
│   └── linear_and_polynomial_test_y.csv
├── models/                    # Pre-trained models
│   ├── linear_regression_demo.mdl
│   └── polynomial_regression_demo.mdl
├── output/                    # Output image
│   └── Linear and Polynomial Visualisation Demo.jpg
├── visualise_regression.py    # Main script
└── README.md
```

## How to run

1. Make sure you have Python installed with the following libraries:
   - pandas
   - numpy
   - matplotlib
   - scikit-learn

2. Run the script:

   ```
   python visualise_regression.py
   ```

3. The output image will be saved in the `output/` folder as:

   ```
   Linear and Polynomial Visualisation Demo.jpg
   ```

## Notes

- The polynomial model uses transformed input features (e.g., x², x³), but the visualisation uses the original X values to make the comparison fair.
- Input values are sorted before plotting to ensure clean and smooth regression lines.









