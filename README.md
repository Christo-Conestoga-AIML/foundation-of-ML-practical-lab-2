# Multivariate Linear Regression, Non-Parametric Models and Cross-Validation

**Author**: Christo Pananjickal Baby  
**Course**: Applied Artificial Intelligence and Machine Learning  
**Assignment**: Practical Lab 2  
**Date**: 18-06-2025  

## Dataset
This assignment uses the Diabetes dataset provided by Scikit-Learn. It contains medical and physiological data of patients, including BMI, age, blood pressure, and more, along with a target indicating diabetes disease progression.

## Objective
To build, evaluate and find out best models that predict the risk of diabetes progression.

## Approach
The assignment is divided into three parts: univariate polynomial regression, multivariate regression and non-parametric models. Models are evaluated using R-squared, MAE, and MAPE across train, validation, and test sets with proper visualizations and analysis.

## Object-Oriented Programming (OOP)
A reusable `ModelEvaluator` class is designed to encapsulate all model evaluation steps, making it easy to switch between models and features. This modular structure improves code readability, maintenance, and scalability.