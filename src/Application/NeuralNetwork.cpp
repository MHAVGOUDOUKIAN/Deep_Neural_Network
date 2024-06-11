#include <Application/NeuralNetwork.hpp>

#include<unistd.h>

NeuralNetwork::NeuralNetwork(const int nb_features, const LossFunction loss_type) : nb_features(nb_features) {
    setLossFunction(loss_type);
}

NeuralNetwork::~NeuralNetwork() {}

void NeuralNetwork::addLayer(const int nb_neuron, const Activation activation_type) {
    if(m_bias.size()==0 && m_weights.size()==0) { 
        m_weights.push_back(Matrix(nb_neuron, nb_features)); 
        m_bias.push_back(Matrix(nb_neuron,1));
    } else {
        m_weights.push_back(Matrix(nb_neuron,m_weights[m_weights.size()-1].row())); 
        m_bias.push_back(Matrix(nb_neuron,1));
    }
    m_activ_func.push_back(activation_type);
}

void NeuralNetwork::clear(void) {
    m_bias.clear();
    m_weights.clear();
    m_activ_func.clear();
}

void NeuralNetwork::train(const Matrix& X_train, const Matrix& Y_train, const int epoch, const float learning_rate, const bool show_result) {
    std::cout << "Training started" << std::endl;
    std::vector<Matrix> activ;
    activ.reserve(m_weights.size()+1); // equal to layers number
    activ.push_back(X_train);  
    for(int i=0; i<m_weights.size(); i++) {
        activ.push_back(Matrix(m_weights[i].row(),X_train.col()));
    }

    // Starting training process
    for(int iter=0; iter<epoch; iter++) {    
        // Forward propagation
        for(int i=0; i<m_weights.size(); i++) {
            Matrix Z {m_weights[i]*activ[i]};
            Z.merge(m_bias[i]);
            Z.applySigmo();
            activ[i+1] = Z;
        }

        // Calc lost function
        Matrix temp {Matrix(0,0)};
        if(show_result) {
            Matrix temp_A{activ[activ.size()-1]};
            temp_A*(-1);
            temp_A+1;
            temp_A.applyLog();

            Matrix temp_Y{Y_train};
            temp_Y*(-1);
            temp_Y+1;

            Matrix res{Hadamard(temp_A,temp_Y)};

            Matrix temp{activ[activ.size()-1]};
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
        Matrix dZ{activ[activ.size()-1]};
        temp = Y_train;
        temp*(-1);
        dZ = dZ + temp;

        for(int i=activ.size()-1; i>0; i--) {
            Matrix dW{dZ};
            dW * m;
            dW = dW * (activ[i-1].transposee());

            Matrix dB{SumOnCol(dZ)};
            dB * m;

            dZ = m_weights[i-1].transposee() * dZ;
            temp = activ[i-1];
            temp*(-1);
            temp+1;
            temp= Hadamard(temp,activ[i-1]);
            dZ = Hadamard(dZ, temp); // dZ for the new loop

            dW * (-learning_rate);
            dB * (-learning_rate);
            m_weights[i-1] = m_weights[i-1] + dW;
            m_bias[i-1] = m_bias[i-1] + dB;
        }
    }  
    std::cout << "Training done" << std::endl;
}

void NeuralNetwork::newtrain(const Matrix& X_train, const Matrix& Y_train, const int epoch, const float learning_rate, const bool show_result) {
    std::cout << "Training started" << std::endl;
    // Creating vector to store activation matrix and z Matrix
    std::vector<Matrix> activ;
    std::vector<Matrix> function_z;
    activ.push_back(X_train);  
    function_z.push_back(X_train);  

    float m = 1.f/(double)X_train.col();

    activ.reserve(m_weights.size()+1);
    function_z.reserve(m_weights.size()+1);
    for(int i=0; i<m_weights.size(); i++) {
        activ.push_back(Matrix(m_weights[i].row(),X_train.col()));
        function_z.push_back(Matrix(m_weights[i].row(),X_train.col()));
    }

    // Starting training process
    for(int iter=0; iter<epoch; iter++) {    
        // Forward propagation
        for(int i=0; i<m_weights.size(); i++) {
            function_z[i+1] = m_weights[i]*activ[i];
            function_z[i+1].merge(m_bias[i]);
            activ[i+1] = function_z[i+1];   
            setActivationFunction(m_activ_func[i]);
            activ[i+1] = (*this.*activation)(activ[i+1]);
        }

        // Calc lost function
        Matrix glob_loss_function= (*this.*Loss)(activ[activ.size()-1], Y_train);
        Matrix res = Matrix(Y_train.row(), 1);
        for(int i=0; i<glob_loss_function.row();i++) {
            float add = 0.f;
            for(int j=0; j<glob_loss_function.col();j++) { 
                add += glob_loss_function.getCoeff(i,j);
            } 
            res.setCoeff(i,0,add);
        }   
        res*(-m);
        res.disp();
        std::cout << (float(iter)/float(epoch))*100.f << "%" << std::endl;

        // Back propagation
        Matrix dZ = (*this.*LossDerivate)(activ[activ.size()-1], Y_train);
        dZ = Hadamard(dZ, SigmoidDerivate(function_z[function_z.size()-1]));

        for(int i=activ.size()-1; i>0; i--) {
            Matrix dW{dZ};
            dW*m;
            dW = dW*(activ[i-1].transposee());

            Matrix dB{SumOnCol(dZ)};
            dB * m;

            dZ = m_weights[i-1].transposee() * dZ;
            dZ = Hadamard(dZ, (*this.*activationDerivate)(function_z[i-1]));

            dW * (-learning_rate);
            dB * (-learning_rate);
            
            m_weights[i-1] = m_weights[i-1] + dW;
            m_bias[i-1] = m_bias[i-1] + dB;
        }
    }

    std::cout << "Training done" << std::endl;
}

Matrix NeuralNetwork::predict(const Matrix& X_test, const Matrix& Y_test) {
    std::cout << "PREDICTIONS" << std::endl;
    Matrix predict {X_test};
    for(int i=0; i<m_weights.size(); i++) {
            predict = m_weights[i]*predict;
            predict.merge(m_bias[i]);
            setActivationFunction(m_activ_func[i]);
            predict = (*this.*activation)(predict);
    }
    X_test.disp();
    predict.disp();
    Y_test.disp();
    return predict;
}

void NeuralNetwork::setLossFunction(LossFunction type) {
    switch(type) {
        case LossFunction::BINARY_CROSS_ENTROPY:
            Loss = &NeuralNetwork::LogLoss;
            LossDerivate = &NeuralNetwork::LogLossDerivate;
            break;
    }
}

void NeuralNetwork::setActivationFunction(Activation type) {
    switch(type) {
        case Activation::SIGMOID:
            activation = &NeuralNetwork::Sigmoid;
            activationDerivate = &NeuralNetwork::SigmoidDerivate;
            break;
        
        case Activation::ELU:
            activation = &NeuralNetwork::ELU;
            activationDerivate = &NeuralNetwork::ELUDerivate;
            break;
        
        case Activation::RELU:
            activation = &NeuralNetwork::RELU;
            activationDerivate = &NeuralNetwork::RELUDerivate;
            break;
    }
}

//////////////
// LOSS FUNCTION AND DERIVATE
//////////////

// Apply Log loss on matrix A and  Y
// NOTE: dim(A) = dim(Y)
Matrix NeuralNetwork::LogLoss(const Matrix& A, const Matrix& Y) {
    //A.disp();
    float eps = 1e-15;
    Matrix res = Matrix(A.row(), A.col(),0.f);
    for(int i=0; i<A.row(); i++) {
        for(int j=0; j<A.col(); j++) {
            res.setCoeff(i,j, Y.getCoeff(i,j) * log(A.getCoeff(i,j)+eps)+ (1.f-Y.getCoeff(i,j))*log(1.f-A.getCoeff(i,j)+eps));
        }   
    }
    return res;
}

// Apply Log loss derivate on matrix A and  Y
// NOTE: dim(A) = dim(Y)
Matrix NeuralNetwork::LogLossDerivate(const Matrix& A, const Matrix& Y) {
    float eps = 1e-15;
    Matrix res {Matrix(A.row(), A.col(),0.f)};
    for(int i=0; i<A.row(); i++) {
        for(int j=0; j<A.col(); j++) {
            res.setCoeff(i,j, -Y.getCoeff(i,j)/(A.getCoeff(i,j)+eps) + (1.f-Y.getCoeff(i,j))/(1.f-A.getCoeff(i,j)+eps));
        }   
    }
    return res;
}

//////////////
// ACTIVATION FUNCTION AND DERIVATE
//////////////

Matrix NeuralNetwork::Sigmoid(const Matrix& Z) {
    Matrix res {Matrix(Z.row(), Z.col(),0.f)};
    for(int i=0; i<Z.row(); i++) {
        for(int j=0; j<Z.col(); j++) {
            res.setCoeff(i,j, 1.f / (1.f + exp(-1.f * Z.getCoeff(i,j))));
        }
    }
    return res;
}

Matrix NeuralNetwork::SigmoidDerivate(const Matrix& Z) {
    Matrix res {Matrix(Z.row(), Z.col(),0.f)};
    for(int i=0; i<Z.row(); i++) {
        for(int j=0; j<Z.col(); j++) {
            float sigmo = 1.f / (1.f + exp(-1.f * Z.getCoeff(i,j)));
            res.setCoeff(i,j, sigmo * (1.f - sigmo));
        }
    }
    return res;
}

Matrix NeuralNetwork::ELU(const Matrix& Z) {
    float alpha=1.f;
    Matrix res {Z};
    for(int i=0; i<Z.row(); i++) {
        for(int j=0; j<Z.col(); j++) {
            if(Z.getCoeff(i,j)>0.f) res.setCoeff(i,j, Z.getCoeff(i,j));
            else res.setCoeff(i,j,alpha*(exp(Z.getCoeff(i,j))-1.f));
        }
    }
    return res;
}

Matrix NeuralNetwork::ELUDerivate(const Matrix& Z) {
    float alpha=1.f;
    Matrix res {Z};
    for(int i=0; i<Z.row(); i++) {
        for(int j=0; j<Z.col(); j++) {
            if(Z.getCoeff(i,j)>=0.f) res.setCoeff(i,j, 1.f);
            else res.setCoeff(i,j,alpha*exp(Z.getCoeff(i,j)));
        }
    }
    return res;
}

Matrix NeuralNetwork::RELU(const Matrix& Z) {
    Matrix res {Z};
    for(int i=0; i<Z.row(); i++) {
        for(int j=0; j<Z.col(); j++) {
            if(Z.getCoeff(i,j)>0.f) res.setCoeff(i,j, Z.getCoeff(i,j));
            else res.setCoeff(i,j,0.f);
        }
    }
    return res;
}

Matrix NeuralNetwork::RELUDerivate(const Matrix& Z) {
    Matrix res {Z};
    for(int i=0; i<Z.row(); i++) {
        for(int j=0; j<Z.col(); j++) {
            if(Z.getCoeff(i,j)>0.f) res.setCoeff(i,j, 1.f);
            else res.setCoeff(i,j,0.f);
        }
    }
    return res;
}