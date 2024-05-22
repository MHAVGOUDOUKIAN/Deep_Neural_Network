#include <Application/NeuralNetwork.hpp>

#include<unistd.h>
unsigned int microsecond = 1000000;


NeuralNetwork::NeuralNetwork(const int nb_features) : nb_features(nb_features) {}

NeuralNetwork::~NeuralNetwork() {}

void NeuralNetwork::addLayer(const int nb_neuron) {
    if(m_bias.size()==0 && m_weights.size()==0) { 
        m_weights.push_back(Matrix(nb_neuron, nb_features)); 
        m_bias.push_back(Matrix(nb_neuron,1));
    } else {
        m_weights.push_back(Matrix(nb_neuron,m_weights[m_weights.size()-1].row())); 
        m_bias.push_back(Matrix(nb_neuron,1));
    }
}

void NeuralNetwork::clear(void) {
    m_bias.clear();
    m_weights.clear();
}

void NeuralNetwork::train(const Matrix& X_train, const Matrix& Y_train, const int epoch, const float learning_rate, const bool show_result) {
    std::cout << "Training started" << std::endl;
    std::vector<Matrix> activation;
    activation.reserve(m_weights.size()+1); // equal to layers number
    activation.push_back(X_train);  
    for(int i=0; i<m_weights.size(); i++) {
        activation.push_back(Matrix(m_weights[i].row(),X_train.col()));
    }

    // Strating training process
    for(int iter=0; iter<epoch; iter++) {    
        // Forward propagation
        for(int i=0; i<m_weights.size(); i++) {
            Matrix Z {m_weights[i]*activation[i]};
            Z.merge(m_bias[i]);
            Z.applySigmo();
            activation[i+1] = Z;
        }

        // Calc lost function
        Matrix temp {Matrix(0,0)};
        if(show_result) {
            Matrix temp_A{activation[activation.size()-1]};
            temp_A*(-1);
            temp_A+1;
            temp_A.applyLog();

            Matrix temp_Y{Y_train};
            temp_Y*(-1);
            temp_Y+1;

            Matrix res{Hadamard(temp_A,temp_Y)};

            Matrix temp{activation[activation.size()-1]};
            temp.applyLog();
            temp = Hadamard(Y_train,temp);
            res = res+temp;

            Matrix loss { Matrix(res.row(),1)};
            
            for(int i=0; i<res.row();i++) {
                for(int j=0; j<res.col();j++) loss.setCoeff(i,0,res.getCoeff(0,i));
                loss.setCoeff(i,0, loss.getCoeff(i,0) * -(1.f/(double)X_train.col()));
            }

            loss.disp();
            std::cout << (float(iter)/float(epoch))*100.f << "%" << std::endl;
        }

        // Back propagation
        float m = 1.f/(float)X_train.col();
        Matrix dZ{activation[activation.size()-1]};
        temp = Y_train;
        temp*(-1);
        dZ = dZ + temp;

        for(int i=activation.size()-1; i>0; i--) {
            Matrix dW{dZ};
            dW * m;
            dW = dW * (activation[i-1].transposee());

            Matrix dB{SumOnCol(dZ)};
            dB * m;

            dZ = m_weights[i-1].transposee() * dZ;
            temp = activation[i-1];
            temp*(-1);
            temp+1;
            temp= Hadamard(temp,activation[i-1]);
            dZ = Hadamard(dZ, temp); // dZ for the new loop

            dW * (-learning_rate);
            dB * (-learning_rate);
            m_weights[i-1] = m_weights[i-1] + dW;
            m_bias[i-1] = m_bias[i-1] + dB;
        }
    }  
    std::cout << "Training done" << std::endl;
}

void NeuralNetwork::predict(const Matrix& X_test, const Matrix& Y_test) {
    std::cout << "PREDICTIONS" << std::endl;
    Matrix predict {X_test};
    for(int i=0; i<m_weights.size(); i++) {
            predict = m_weights[i]*predict;
            predict.merge(m_bias[i]);
            predict.applySigmo();
    }
    X_test.disp();
    predict.disp();
    Y_test.disp();
}