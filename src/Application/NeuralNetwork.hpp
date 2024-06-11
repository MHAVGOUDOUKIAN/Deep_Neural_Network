#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <Application/Matrix.hpp>
#include <vector>

enum Activation {SIGMOID, RELU, ELU};
enum LossFunction {BINARY_CROSS_ENTROPY};

class NeuralNetwork {
    public:
        NeuralNetwork(const int number_of_features, const LossFunction loss_type);
        ~NeuralNetwork();

        void addLayer(const int nb_neuron, const Activation activation_type = Activation::SIGMOID);
        void clear(void);
        void train(const Matrix& X_train, const Matrix& Y_train, const int epoch=1, const float learning_rate=1.0f, const bool show_result=true);
        void newtrain(const Matrix& X_train, const Matrix& Y_train, const int epoch=1, const float learning_rate=1.0f, const bool show_result=true);
        Matrix predict(const Matrix& X_test, const Matrix& Y_test);

    private:
        // CHANGE ACTIVATION AND LOSSFUNCTION
        void setLossFunction(LossFunction type);
        void setActivationFunction(Activation type);

        // LOSS FUNCTION AND DERIVATE
        Matrix LogLoss(const Matrix& A, const Matrix& Y);
        Matrix LogLossDerivate(const Matrix& A, const Matrix& Y);

        // ACTIVATION FUNCTION AND DERIVATE
        Matrix Sigmoid(const Matrix& Z);
        Matrix SigmoidDerivate(const Matrix& Z) ;

        Matrix ELU(const Matrix& Z);
        Matrix ELUDerivate(const Matrix& Z) ;

        Matrix RELU(const Matrix& Z);
        Matrix RELUDerivate(const Matrix& Z) ;

    private:
        std::vector<Matrix> m_weights;
        std::vector<Matrix> m_bias;
        std::vector<Activation> m_activ_func;
        int nb_features;

        // Functions called when calculating activation or loss
        Matrix (NeuralNetwork::*activation) (const Matrix&);
        Matrix (NeuralNetwork::*activationDerivate) (const Matrix&);
        Matrix (NeuralNetwork::*Loss) (const Matrix&, const Matrix&);
        Matrix (NeuralNetwork::*LossDerivate) (const Matrix&, const Matrix&);
};



#endif