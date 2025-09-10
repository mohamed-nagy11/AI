# Simple Linear Regression for TV Marketing vs. Sales

This notebook demonstrates how to build and evaluate a simple linear regression model to predict product sales based on TV marketing expenditure.

## Project Description

The goal of this project is to explore the relationship between the cost spent on TV advertising and the resulting sales. A simple linear regression model is implemented using `scikit-learn` to quantify this relationship and make predictions.

## Dataset

The dataset used in this notebook is `tvmarketing.csv`. It contains two columns:
- `TV`: The cost spent on TV advertising.
- `Sales`: The resulting sales.

The dataset is uploaded to: `https://github.com/mohamed-nagy11/AI/blob/main/Linear%20Regression/Sales%20Based%20on%20TV%20Marketing/tvmarketing.csv`.

## Notebook Structure

1.  **Drive Mount**: Connects Google Colab to Google Drive to access the dataset.
2.  **Data Exploration**: Loads the dataset, displays basic information (head, tail, info, shape, describe), and visualizes the relationship between TV marketing cost and Sales using scatter plots.
3.  **Linear Regression**: Prepares the data by splitting it into training and testing sets and reshaping the feature data for compatibility with scikit-learn.
4.  **Fitting a Linear Regression Model**: Trains a simple linear regression model on the training data.
5.  **Predictions**: Makes predictions on the unseen test data.
6.  **Evaluation**: Calculates and displays evaluation metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) to assess the model's performance.
7.  **Visualization**: Visualizes the actual vs. predicted sales and the error term. Also plots the training data, testing data, and the fitted regression line.

## How to Run the Notebook

1.  **Clone the Repository**: Clone this GitHub repository to your local machine or access it directly through Google Colab.
2.  **Open in Colab**: Open the notebook (`your_notebook_name.ipynb`) in Google Colab.
3.  **Mount Google Drive**: Run the "Drive Mount" cell to connect to your Google Drive. Then, change the directory to the folder containing the dataset `tvmarketing.csv`.
4.  **Run Cells**: Execute the cells sequentially to perform data loading, exploration, model training, prediction, and evaluation.

## Dependencies

The notebook requires the following libraries, which are pre-installed in Google Colab:
- pandas
- numpy
- matplotlib
- scikit-learn

## Results

The notebook provides the calculated coefficients of the linear regression model (intercept and slope), as well as the MSE, RMSE, and R2 score on the test set. Visualizations help to understand the model's fit and the distribution of errors.

## Recommendations

*   Explore feature transformations (e.g., logarithmic) to address non-linearity and heteroscedasticity.
*   Consider polynomial regression to capture potential curved relationships.
*   Evaluate other regression models (e.g., Decision Trees, Random Forests, SVR) and compare their performance.
*   Perform cross-validation for more robust model evaluation.
