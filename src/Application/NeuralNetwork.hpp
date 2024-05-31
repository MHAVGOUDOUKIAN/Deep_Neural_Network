#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <Application/Matrix.hpp>
#include <vector>

class NeuralNetwork {
    public:
        NeuralNetwork(const int number_of_features);
        ~NeuralNetwork();

        void addLayer(const int nb_neuron);
        void clear(void);
        void train(const Matrix& X_train, const Matrix& Y_train, const int epoch=1, const float learning_rate=1.0f, const bool show_result=true);
        void newtrain(const Matrix& X_train, const Matrix& Y_train, const int epoch=1, const float learning_rate=1.0f, const bool show_result=true);
        Matrix predict(const Matrix& X_test, const Matrix& Y_test);

    private:
        Matrix applyLogLoss(const Matrix& A, const Matrix& Y);
        Matrix applyLogLossDerivate(const Matrix& A, const Matrix& Y);

    private:
        std::vector<Matrix> m_weights;
        std::vector<Matrix> m_bias;
        int nb_features;
};

// SIGMOID FUNCTION
Matrix calculateSigmActivation(Matrix& Z);
Matrix derivateSigm(Matrix& Z) ;
#endif