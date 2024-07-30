#include <vector>

using namespace std;

class LinearRegression {
private:
    vector<double> weights;

public:
    LinearRegression(int numFeatures);

    void fit(const vector<vector<double>>& X_train, const vector<double>& y_train, double learningRate = 0.001, int epochs = 1000);
    double predict(const vector<double>& features);
};

// Member function definitions

LinearRegression::LinearRegression(int numFeatures) {
    // Initialize weights with zeros
    weights = vector<double>(numFeatures + 1, 0.0); // Additional weight for bias
}

void LinearRegression::fit(const vector<vector<double>>& X_train, const vector<double>& y_train, double learningRate, int epochs) {
    int numSamples = X_train.size();
    int numFeatures = X_train[0].size();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < numSamples; ++i) {
            double prediction = predict(X_train[i]);
            double error = y_train[i] - prediction;

            // Update weights and bias
            for (int j = 0; j < numFeatures; ++j) {
                weights[j] += learningRate * error * X_train[i][j];
            }
            // Update bias weight
            weights[numFeatures] += learningRate * error;
        }
    }
}

double LinearRegression::predict(const vector<double>& features) {
    double result = weights[features.size()];
    for (int i = 0; i < features.size(); ++i) {
        result += weights[i] * features[i];
    }
    return result;
}