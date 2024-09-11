#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

class LinearRegression {
public:
    // Fit the model to the dataset
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        assert(X.size() == y.size() && "Features and labels size mismatch");
        size_t m = X.size();   // Number of training examples
        size_t n = X[0].size(); // Number of features

        // Initialize coefficients (weights) and intercept
        coefficients = std::vector<double>(n, 0.0);
        intercept = 0.0;

        // Gradient descent parameters
        double learning_rate;
        size_t epochs;
        std::cout<<"Write the learning rate"<<std::endl;
        std::cin>> learning_rate;
        std::cout<<"Write the size of the epochs"<<std::endl;
        std::cin>> epochs;

        // Training using Gradient Descent
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            std::vector<double> gradients(n, 0.0);
            double intercept_gradient = 0.0;

            // Calculate gradients
            for (size_t i = 0; i < m; ++i) {
                double prediction = predict(X[i]);
                double error = prediction - y[i];
                intercept_gradient += error;
                for (size_t j = 0; j < n; ++j) {
                    gradients[j] += error * X[i][j];
                }
            }

            // Update coefficients and intercept
            intercept -= learning_rate * intercept_gradient / m;
            for (size_t j = 0; j < n; ++j) {
                coefficients[j] -= learning_rate * gradients[j] / m;
            }
        }
    }

    // Predict using the fitted model
    double predict(const std::vector<double>& x) const {
        double prediction = intercept;
        for (size_t j = 0; j < coefficients.size(); ++j) {
            prediction += coefficients[j] * x[j];
        }
        return prediction;
    }

private:
    std::vector<double> coefficients;
    double intercept;
};

int main() {
    // Example dataset
    std::vector<std::vector<double>> X = {
        {1, 2},
        {2, 3},
        {3, 4}
    };
    std::vector<double> y = {3, 5, 7};

    // Create and train the linear regression model
    LinearRegression model;
    model.fit(X, y);

    // Make predictions
    for (const auto& features : X) {
        std::cout << "Prediction: " << model.predict(features) << std::endl;
    }

    return 0;
}