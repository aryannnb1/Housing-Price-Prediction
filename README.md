# Housing Price Prediction

A machine learning project to predict house prices in Mumbai based on key real estate features. It includes data preprocessing, exploratory data analysis (EDA), and model training using regression algorithms to achieve accurate predictions.

## Overview

This project aims to predict house prices in Mumbai based on various features such as location, number of bedrooms, square footage, and more. The dataset is cleaned, preprocessed, and analyzed before training a machine learning model to make accurate price predictions.

## Dataset

The dataset contains information on Mumbai real estate properties with the following key attributes:

- **Location**: The area in Mumbai where the property is located.
- **Size**: Number of bedrooms.
- **Total Square Feet**: Total area of the property.
- **Price**: The actual price of the property.
- **Price per Square Foot**: Derived feature to normalize price comparisons.

## Data Preprocessing

### Handling Missing Values
- Dropped rows with critical missing data.
- Imputed missing values where necessary.

### Feature Engineering
- Created a new feature: Price per Square Foot.
- Reduced the cardinality of categorical variables.

### Data Normalization & Cleaning
- Converted inconsistent formats in price values.
- Removed outliers based on statistical analysis.

## Exploratory Data Analysis (EDA)

- Analyzed price distributions and outliers.
- Examined correlations between features.
- Visualized price trends across different locations.

## Model Training

Utilized regression models to predict house prices. Tested various algorithms, including:

- **Linear Regression**
- **Decision Trees**
- **Random Forest**
- **Gradient Boosting Regressor**

Evaluated model performance using RMSE and R² scores.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Jupyter Notebook or any Python IDE

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aryannnb1/Housing-Price-Prediction.git
   cd Housing-Price-Prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Run the data preprocessing and EDA notebook to clean and analyze the dataset.
3. Run the model training notebook to train and evaluate the regression models.

### Results

The final model performance will be evaluated using RMSE and R² scores to determine the accuracy and reliability of the predictions.
