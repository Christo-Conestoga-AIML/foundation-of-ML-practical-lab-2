from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

class ModelEvaluator:
    def __init__(self, train_df, valid_df, test_df, model, features=None, target='target'):
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.model = model
        self.features = features if features else ['bmi']  # default to ['bmi']
        self.target = target
        self.train_valid_results = None
        self.test_results = None
        self.train_val_df = None
        self.poly = None

    def evaluate_on_train_valid_set(self, degrees=None):
        if degrees is None:
            degrees = range(6)  # default degrees 0 to 5

        x_train_bmi = self.train_df[['bmi']].values
        y_train = self.train_df['target'].values
        x_val_bmi = self.valid_df[['bmi']].values
        y_val = self.valid_df['target'].values

        results = []

        for degree in degrees:
            # Polynomial transformation
            poly = PolynomialFeatures(degree=degree, include_bias=True)
            x_train_poly = poly.fit_transform(x_train_bmi)
            x_val_poly = poly.transform(x_val_bmi)

            # Fit model
            self.model.fit(x_train_poly, y_train)

            # Predict
            y_train_pred = self.model.predict(x_train_poly)
            y_val_pred = self.model.predict(x_val_poly)

            # Evaluate
            train_r2 = r2_score(y_train, y_train_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_mape = mean_absolute_percentage_error(y_train, y_train_pred)

            val_r2 = r2_score(y_val, y_val_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            val_mape = mean_absolute_percentage_error(y_val, y_val_pred)

            results.append({
                "Degree": degree,
                "Train R2": train_r2,
                "Train MAE": train_mae,
                "Train MAPE": train_mape,
                "Val R2": val_r2,
                "Val MAE": val_mae,
                "Val MAPE": val_mape
            })

        self.train_valid_results = pd.DataFrame(results)

    def print_train_valid_results(self):
        self.train_valid_results.rename(columns={
            "Degree": "Degree",
            "Train R2": "Train R-Squared",
            "Train MAE": "Train MAE",
            "Train MAPE": "Train MAPE",
            "Val R2": "Val R-Squared",
            "Val MAE": "Val MAE",
            "Val MAPE": "Val MAPE"
        }, inplace=True)
        print(tabulate(self.train_valid_results, headers='keys', tablefmt='psql', floatfmt=".4f"))

    def select_best_model(self):
        sorted_df = self.train_valid_results.sort_values(
            by=['Val R-Squared', 'Val MAE', 'Val MAPE'],
            ascending=[False, True, True]
        ).reset_index(drop=True)
        best_model = sorted_df.iloc[0]
        print(f"Best model found with degree {best_model['Degree']}:")
        return best_model

    def evaluate_on_test_set(self, best_degree):
        self.train_val_df = pd.concat([self.train_df, self.valid_df], axis=0)
        x_train_val = self.train_val_df[self.features].values
        y_train_val = self.train_val_df[self.target].values
        x_test = self.test_df[self.features].values
        y_test = self.test_df[self.target].values

        self.poly = PolynomialFeatures(degree=int(best_degree), include_bias=True)
        x_train_val_poly = self.poly.fit_transform(x_train_val)
        x_test_poly = self.poly.transform(x_test)

        self.model.fit(x_train_val_poly, y_train_val)
        y_test_pred = self.model.predict(x_test_poly)

        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

        self.test_results = pd.DataFrame([{
            "Degree": int(best_degree),
            "Test R-Squared": test_r2,
            "Test MAE": test_mae,
            "Test MAPE": test_mape
        }])

    def print_test_results(self):
        print(tabulate(self.test_results, headers='keys', tablefmt='psql', floatfmt=".4f"))

    def plot_model_fit(self, best_degree):
        if len(self.features) != 1:
            print("Plotting is only available for univariate models.")
            return

        feature = self.features[0]
        bmi_range = np.linspace(min(self.train_val_df[feature].min(), self.test_df[feature].min()),
                                max(self.train_val_df[feature].max(), self.test_df[feature].max()), 300).reshape(-1, 1)
        bmi_range_poly = self.poly.transform(bmi_range)
        y_curve = self.model.predict(bmi_range_poly)

        plt.figure(figsize=(10, 6))
        plt.scatter(self.train_df[feature], self.train_df[self.target], label='Train Data', color='blue', alpha=0.6)
        plt.scatter(self.valid_df[feature], self.valid_df[self.target], label='Validation Data', color='green', alpha=0.6)
        plt.scatter(self.test_df[feature], self.test_df[self.target], label='Test Data', color='red', alpha=0.6)
        plt.plot(bmi_range, y_curve, label=f'Polynomial Fit (Degree {best_degree})', color='black', linewidth=2)

        plt.xlabel(feature)
        plt.ylabel(self.target)
        plt.title(f"Polynomial Regression Fit (Degree {best_degree})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def print_model_equation(self):
        coefs = self.model.coef_
        intercept = self.model.intercept_
        terms = self.poly.get_feature_names_out(self.features)

        equation = f"y = {intercept:.2f}"
        for coef, term in zip(coefs, terms):
            if term == '1':
                continue
            sign = " + " if coef >= 0 else " - "
            equation += f"{sign}{abs(coef):.2f}*{term}"

        print(f"Model Equation: {equation}")

    def predict_for_feature(self, feature_dict):
        # Check keys
        expected_features = set(self.features)
        input_features = set(feature_dict.keys())

        missing = expected_features - input_features
        extra = input_features - expected_features

        if missing:
            print(f"Error: Missing feature(s): {', '.join(missing)}")
            return
        if extra:
            print(f"Warning: Extra feature(s) ignored: {', '.join(extra)}")

        # Order feature values in the same order as self.features
        feature_values = [feature_dict[feat] for feat in self.features]

        feature_input = np.array([feature_values])  # shape (1, n_features)
        feature_poly = self.poly.transform(feature_input)
        prediction = self.model.predict(feature_poly)[0]

        features_str = ", ".join([f"{feat}={val:.4f}" for feat, val in zip(self.features, feature_values)])

        print(f"Predicted {self.target} for {features_str} is: {prediction:.4f}")

    def print_num_trainable_params(self):
        feature_names = self.poly.get_feature_names_out(self.features)
        num_params = len(feature_names)
        print(f"Trainable parameters (including bias/intercept): {num_params}")
        print("Features used in the model:")
        for name in feature_names:
            print(f"  - {name}")
