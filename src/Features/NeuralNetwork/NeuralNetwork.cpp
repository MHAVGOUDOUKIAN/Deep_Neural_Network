#include <Features/NeuralNetwork/NeuralNetwork.hpp>

NeuralNetwork::NeuralNetwork(const int nb_entries) : m_nb_entries(nb_entries) {
    clear();
}

void NeuralNetwork::addLayer(const int nb_neuron, const ActivationType type) {
    if(l_bias.size()==0) { 
        l_weight.push_back(Matrix(nb_neuron,m_nb_entries)); 
        l_bias.push_back(Matrix(nb_neuron,1));
        l_activationFunc.push_back(type);
    } else {
        l_weight.push_back(Matrix(nb_neuron,l_weight[l_weight.size()-1].row())); 
        l_bias.push_back(Matrix(nb_neuron,1));
        l_activationFunc.push_back(type);
    }
    return;
}

void NeuralNetwork::clear(void) {
    l_bias.clear();
    l_weight.clear();
    l_activationFunc.clear();
    return;
}

void NeuralNetwork::train(const Matrix& X_train, const Matrix& Y_train, const int epoch, const float learning_rate) {
    std::vector<Matrix> l_layerZ;
    std::vector<Matrix> l_estimationA;

    std::cout << "Training started ~~~~~~~" << std::endl;
    for(int e=0; e<epoch; e++) {
        // Initialisation
        /* X_train correspond à A0*/
        l_estimationA.push_back(X_train);

        // Forward propagation
        for(int i=0; i<l_weight.size(); i++) {
            Matrix Z=l_weight[i]*l_estimationA[i];
            Z = BroadCastAdd(Z, l_bias[i]);
            l_layerZ.push_back(Z);
            Z.applySigmo();
            l_estimationA.push_back(Z);
        }

        //Back propagation
        Matrix dZ = l_estimationA[l_estimationA.size()-1];
        dZ-=Y_train;
        
        for(int i=l_weight.size()-1; i>=0; i--) {
            Matrix dW = dZ*l_estimationA[i].transposee();
            dW.constMult(1.0f/l_estimationA[i].row());

            Matrix db = SumOnCol(dZ);
            db.constMult(1.0f/X_train.row());
            
            l_bias[i] -= db;
            l_bias[i].constMult(learning_rate);
            l_weight[i] -= dW;
            l_weight[i].constMult(learning_rate);
            l_bias[i].disp();

            if(i!=0) {
                dZ = l_weight[i].transposee()*dZ;
                Matrix temp = Matrix(l_estimationA[i].row(),l_estimationA[i].col(),1.0f);
                temp-=l_estimationA[i];
                temp = Hadamard(l_estimationA[i],temp);
                dZ = Hadamard(dZ,temp);
            }
        }
    }
    std::cout << "Training finished ~~~~~~~" << std::endl;
    l_estimationA[l_estimationA.size()-1].disp();
    return;
}

Matrix NeuralNetwork::predict(const Matrix& X_test, const Matrix& Y_test) {
    std::vector<Matrix> l_layerZ;
    std::vector<Matrix> l_estimationA;

    std::cout << "Predictions ~~~~~~~" << std::endl;
    // Initialisation
    /* X_train correspond à A0*/
    l_estimationA.push_back(X_test);

    // Forward propagation
    for(int i=0; i<l_weight.size(); i++) {
        Matrix Z=l_weight[i]*l_estimationA[i];
        Z = BroadCastAdd(Z, l_bias[i]);
        l_layerZ.push_back(Z);
        Z.applySigmo();
        l_estimationA.push_back(Z);
    }

    return l_estimationA[l_estimationA.size()-1];
}

void NeuralNetwork::show(void) {
    for(int i=0; i<l_weight.size(); i++) {
        std::cout << "LAYER " << i+1 << " (" << l_weight[i].row() << " neuron(s))" << ": " << std::endl;
        std::cout << "Bias" << std::endl;
        l_bias[i].disp();
        std::cout << "Weight" << std::endl;
        l_weight[i].disp();
        std::cout << std::endl;
    }
    std::cout << m_nb_entries << " entries and " << l_weight.size() << " layer(s)" << std::endl;
    return;
}

void NeuralNetwork::setEntriesNumber(const int nb_entries) {
    m_nb_entries = nb_entries;
    return;
}

NeuralNetwork::~NeuralNetwork() {}