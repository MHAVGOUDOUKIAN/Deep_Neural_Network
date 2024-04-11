#ifndef NN_HPP_INCLUDED
#define NN_HPP_INCLUDED

#include <Features/Matrix/Matrix.hpp>

#include <vector>

enum ActivationType { SIGMOID };

class NeuralNetwork {
    public:
        NeuralNetwork(const int nb_entries=2);
        ~NeuralNetwork();

        void addLayer(const int nb_neuron, const ActivationType type = SIGMOID);
        void clear(void);
        /* Represents the number of attributes of your dataset */
        void setEntriesNumber(const int nb_entries);

        /* Train the neural network with the given dataset.
        
            X_Train.row = number of attributes of your data ( or the number of entries of your NN)
            X_Train.col = number of data 

            Y_train.row = number of neurons on the last layer of your NN
            Y_train.col = number of data 
        */
        void train(const Matrix& X_train, const Matrix& Y_train, const int epoch=1, const float learning_rate=1.0f);
        Matrix predict(const Matrix& X_train, const Matrix& Y_train);
        void show(void);

    private:
        std::vector<ActivationType> l_activationFunc;
        std::vector<Matrix> l_weight;
        std::vector<Matrix> l_bias;

        int m_nb_entries;
};

#endif