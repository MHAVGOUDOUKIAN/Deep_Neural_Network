#include <Application/App.hpp>

#define log(str) std::cout << str << std::endl;

App::App() {
    EventHandler::getEventHandler()->addKeyBoardObserver(this);
    m_data_coord = sf::VertexArray(sf::Quads,4);
    m_frontier = sf::VertexArray(sf::Lines, 2);
    m_bias.clear();
    m_weights.clear();

    nn = new NeuralNetwork(2);
    nn->addLayer(4);
    nn->addLayer(3);    

    // Generate training data
    int data_number_train{100};
    Matrix X_train{Matrix(2,data_number_train)}, Y_train{Matrix(3,data_number_train)};
    generate_data_3_class(X_train, Y_train,true);

    // Training
    //train_doubleLayer(X_train, Y_train,100000, 0.5f, true);
    nn->train(X_train,Y_train, 10000, 0.3f,true);
}

App::~App() {}

void App::update(sf::Time deltaTime) {

}

void App::notify(sf::Keyboard::Key key, bool pressed) {
    if(key == sf::Keyboard::Space && pressed) {
        // Generate testing data
        int data_number_test{10};
        Matrix X_test{Matrix(2,data_number_test)}, Y_test{Matrix(3,data_number_test)};
        generate_data_3_class(X_test, Y_test, true);

        // Predictions
        //predict(X_test, Y_test);
        nn->predict(X_test, Y_test);
    }
}

void App::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    target.draw(m_data_coord);
    target.draw(m_frontier);
}

// This function generate linearly separable data from 2 class
// X_feature max dim : row=2, col= nb_of_data
// Y_class max dim: row=1, col=nb_of_data
void App::generate_data_linear(Matrix& X_feature, Matrix& Y_class, bool update_graphics) {    
    const float a{-1.4f}, b{1.0f}, c{0.3f};
    const float zoom{500.f};
    sf::Color class_color{sf::Color::Cyan};
    int nb_of_class0=0;
    if(update_graphics) {
        m_data_coord = sf::VertexArray(sf::Quads,X_feature.col()*4);
        m_frontier = sf::VertexArray(sf::Lines, 2);  // Used to draw frontier with SFML
    }

    // Draw frontier between class 0 and class 1
    m_frontier[0].position = sf::Vector2f((-c/a)*zoom,1.0f*zoom);
    m_frontier[1].position = sf::Vector2f(((-(b+c)/a))*zoom,0.f);
    m_frontier[0].color = sf::Color::White;
    m_frontier[1].color = sf::Color::White;

    // Create random data
    for(int i=0; i<X_feature.col(); i++) {
        for(int j=0; j<X_feature.row(); j++) X_feature.setCoeff(j,i,randomf(0,1));
        
        if(update_graphics) {
            m_data_coord[i*4].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom-2,(1.0f-X_feature.getCoeff(1,i))*zoom-2);
            m_data_coord[i*4+1].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom-2,(1.0f-X_feature.getCoeff(1,i))*zoom+2);
            m_data_coord[i*4+2].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom+2,(1.0f-X_feature.getCoeff(1,i))*zoom+2);
            m_data_coord[i*4+3].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom+2,(1.0f-X_feature.getCoeff(1,i))*zoom-2);
        }

        if(X_feature.getCoeff(0,i)*a + X_feature.getCoeff(1,i)*b +c >= 0) { // Class 0
            Y_class.setCoeff(0,i,0);
            class_color = sf::Color::Cyan;
            nb_of_class0++;
        }
        else { // Class 1
            Y_class.setCoeff(0,i,1);
            class_color = sf::Color::Green;
        }

        if(update_graphics) {
            m_data_coord[i*4].color = class_color;
            m_data_coord[i*4+1].color = class_color;
            m_data_coord[i*4+2].color = class_color;
            m_data_coord[i*4+3].color = class_color;
        }
    }
    std::cout << "data created [class 0: " << nb_of_class0 << "(" << ((float)nb_of_class0/(float)X_feature.col())*100.f <<"%)]" << std::endl;
    std::cout << std::setw(23)<< " [class 1: " << X_feature.col() - nb_of_class0 << "(" << ((float)(X_feature.col() - nb_of_class0)/(float)X_feature.col())*100.f <<"%)]" << std::endl;  
}

// This function generate non linearly separable data from 2 class
// X_feature max dim : row=2, col= nb_of_data
// Y_class max dim: row=1, col=nb_of_data
void App::generate_data_circle(Matrix& X_feature, Matrix& Y_class, bool update_graphics) {    
    const float r{0.4f}, x{0.5f}, y{0.5f};
    const float zoom{500.f};
    sf::Color class_color{sf::Color::Cyan};
    int nb_of_class0=0;
    if(update_graphics) {
        m_data_coord = sf::VertexArray(sf::Quads,X_feature.col()*4);
        m_frontier = sf::VertexArray(sf::Lines, 2);  // Used to draw frontier with SFML
    }

    // Create random data
    for(int i=0; i<X_feature.col(); i++) {
        for(int j=0; j<X_feature.row(); j++) X_feature.setCoeff(j,i,randomf(0,1));
        
        if(update_graphics) {
            m_data_coord[i*4].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom-2,(1.0f-X_feature.getCoeff(1,i))*zoom-2);
            m_data_coord[i*4+1].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom-2,(1.0f-X_feature.getCoeff(1,i))*zoom+2);
            m_data_coord[i*4+2].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom+2,(1.0f-X_feature.getCoeff(1,i))*zoom+2);
            m_data_coord[i*4+3].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom+2,(1.0f-X_feature.getCoeff(1,i))*zoom-2);
        }

        if((X_feature.getCoeff(0,i)-x)*(X_feature.getCoeff(0,i)-x) + (X_feature.getCoeff(1,i)-y)*(X_feature.getCoeff(1,i)-y) > r*r) { // Class 0
            Y_class.setCoeff(0,i,0);
            class_color = sf::Color::Cyan;
            nb_of_class0++;
        }
        else { // Class 1
            Y_class.setCoeff(0,i,1);
            class_color = sf::Color::Green;
        }

        if(update_graphics) {
            m_data_coord[i*4].color = class_color;
            m_data_coord[i*4+1].color = class_color;
            m_data_coord[i*4+2].color = class_color;
            m_data_coord[i*4+3].color = class_color;
        }
    }
    std::cout << "data created [class 0: " << nb_of_class0 << "(" << ((float)nb_of_class0/(float)X_feature.col())*100.f <<"%)]" << std::endl;
    std::cout << std::setw(23)<< " [class 1: " << X_feature.col() - nb_of_class0 << "(" << ((float)(X_feature.col() - nb_of_class0)/(float)X_feature.col())*100.f <<"%)]" << std::endl;  
}

// This function generate non linearly separable data from 2 class
// X_feature max dim : row=2, col= nb_of_data
// Y_class max dim: row=3, col=nb_of_data
void App::generate_data_3_class(Matrix& X_feature,Matrix& Y_class, bool update_graphics) {
    const float zoom{500.f};
    sf::Color class_color{sf::Color::Cyan};
    int nb_of_class0=0;
    int nb_of_class1=0;
    int nb_of_class2=0;
    if(update_graphics) {
        m_data_coord = sf::VertexArray(sf::Quads,X_feature.col()*4);
    }

    // Create random data
    for(int i=0; i<X_feature.col(); i++) {
        for(int j=0; j<X_feature.row(); j++) X_feature.setCoeff(j,i,randomf(0,1));
        
        if(update_graphics) {
            m_data_coord[i*4].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom-2,(1.0f-X_feature.getCoeff(1,i))*zoom-2);
            m_data_coord[i*4+1].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom-2,(1.0f-X_feature.getCoeff(1,i))*zoom+2);
            m_data_coord[i*4+2].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom+2,(1.0f-X_feature.getCoeff(1,i))*zoom+2);
            m_data_coord[i*4+3].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom+2,(1.0f-X_feature.getCoeff(1,i))*zoom-2);
        }

        if(X_feature.getCoeff(0,i) > 0.5 && X_feature.getCoeff(1,i) > 0.5) { // Class 0
            Y_class.setCoeff(0,i,1);
            Y_class.setCoeff(1,i,0);
            Y_class.setCoeff(2,i,0);
            class_color = sf::Color::Cyan;
            nb_of_class0++;
        }
        else if (X_feature.getCoeff(0,i) <= 0.5 && X_feature.getCoeff(1,i) > 0.5) { // Class 1
            Y_class.setCoeff(0,i,0);
            Y_class.setCoeff(1,i,1);
            Y_class.setCoeff(2,i,0);
            class_color = sf::Color::Green;
            nb_of_class1++;
        }
        else if (X_feature.getCoeff(1,i) <= 0.5) {
            Y_class.setCoeff(0,i,0);
            Y_class.setCoeff(1,i,0);
            Y_class.setCoeff(2,i,1);
            class_color = sf::Color::Red;
            nb_of_class2++;
        }

        if(update_graphics) {
            m_data_coord[i*4].color = class_color;
            m_data_coord[i*4+1].color = class_color;
            m_data_coord[i*4+2].color = class_color;
            m_data_coord[i*4+3].color = class_color;
        }
    }
    std::cout << "data created [class 0: " << nb_of_class0 << "(" << ((float)nb_of_class0/(float)X_feature.col())*100.f <<"%)]" << std::endl;
    std::cout << std::setw(23)<< " [class 1: " << nb_of_class1 << "(" << ((float)nb_of_class1/(float)X_feature.col())*100.f <<"%)]" << std::endl;  
    std::cout << std::setw(23)<< " [class 2: " << nb_of_class2 << "(" << ((float)nb_of_class2/(float)X_feature.col())*100.f <<"%)]" << std::endl;  
}   

//
void App::train_simpleNeuron(const Matrix& X_train,const Matrix& Y_train, const int epoch, const float learning_rate, const bool show_result) {
    std::cout << "Training started" << std::endl;
    m_weights.clear();
    m_bias.clear();
    m_weights.push_back(Matrix(1,X_train.row()));
    m_bias.push_back(Matrix(1,1, 0.5));
    for(int iter=0; iter<epoch; iter++) {        
        // Forward propagation
        Matrix Z {m_weights[0]*X_train};
        Z.merge(m_bias[0]);
        Matrix A{Z};
        A.applySigmo();

        // Calc lost function
        Matrix temp {Matrix(0,0)};
        if(show_result) {
            Matrix temp_A{A};
            temp_A*(-1);
            temp_A+1;
            temp_A.applyLog();

            Matrix temp_Y{Y_train};
            temp_Y*(-1);
            temp_Y+1;

            Matrix res{Hadamard(temp_A,temp_Y)};

            Matrix temp{A};
            temp.applyLog();
            temp = Hadamard(Y_train,temp);
            res+temp;

            double loss = 0;
            for(int i=0; i<res.col();i++) loss += res.getCoeff(0,i);
            loss*=-(1.f/(double)res.col());

            std::cout << (float(iter)/float(epoch))*100.f << "% loss: " << loss << std::endl;
        }

        // Back propagation
        Matrix dW = Y_train;
        dW*(-1);
        dW = dW+A;
        dW = X_train*dW.transposee();
        dW*(1.f/(float)X_train.col());

        Matrix dB = Y_train;
        dB*(-1);
        dB = dB+A;
        float db_L = 0;
        for(int i=0; i<dB.col();i++) db_L += dB.getCoeff(0,i);
        db_L = db_L*(1.f/(float)dB.col());

        // update weight and bias
        temp = dW.transposee();
        temp*(-learning_rate);
        m_weights[0] = m_weights[0]+temp;

        m_bias[0] + (-1)*(learning_rate * db_L);
    }
    std::cout << "Training done" << std::endl;
}

void App::train_doubleLayer(const Matrix& X_train,const Matrix& Y_train, const int epoch, const float learning_rate, const bool show_result) {
    std::cout << "Training started" << std::endl;
    m_weights.clear();
    m_weights.push_back(Matrix(3,2));
    m_weights.push_back(Matrix(1,3));
    m_bias.clear();
    m_bias.push_back(Matrix(3,1));  
    m_bias.push_back(Matrix(1,1));

    // Initialize activation matrix
    std::vector<Matrix> activation;
    activation.reserve(3); // equal to layers number
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
            res+temp;

            double loss = 0;
            for(int i=0; i<res.col();i++) loss += res.getCoeff(0,i);
            loss*=-(1.f/(double)res.col());

            std::cout << (float(iter)/float(epoch))*100.f << "% loss: " << loss << std::endl;
        }

        // Back propagation
        float m = 1.f/(float)X_train.col();
        Matrix dZ2{Y_train};
        dZ2 *(-1);
        dZ2 = dZ2 + activation[activation.size()-1];

        Matrix dW2{dZ2};
        dW2 * m;
        dW2 = dW2 * (activation[activation.size()-2].transposee());
    
        Matrix dB2{SumOnCol(dZ2)};
        dB2 * m;

        Matrix dZ1 = (m_weights[1].transposee()) * dZ2;
        temp = activation[activation.size()-2];
        temp*(-1);
        temp+1;
        temp = Hadamard(temp,activation[activation.size()-2]);
        dZ1 = Hadamard(dZ1,temp);

        Matrix dW1{dZ1};
        dW1 * m;
        dW1 = dW1 * (activation[activation.size()-3].transposee());

        Matrix dB1{SumOnCol(dZ1)};
        dB1 * m;

        // update weight and bias
        dW1 * (-learning_rate);
        dW2 * (-learning_rate);
        dB1 * (-learning_rate);
        dB2 * (-learning_rate);

        m_weights[0] = m_weights[0] + dW1;
        m_weights[1] = m_weights[1] + dW2;
        m_bias[0] = m_bias[0] + dB1;
        m_bias[1] = m_bias[1] + dB2;

    }  
    std::cout << "Training done" << std::endl;
}

void App::predict(const Matrix& X_test,const Matrix& Y_test) {
    log("PREDICTIONS")
    Matrix predict {X_test};
    for(int i=0; i<m_weights.size(); i++) {
            predict = m_weights[i]*predict;
            predict.merge(m_bias[i]);
            predict.applySigmo();
    }
    X_test.disp();
    predict.disp();
    Y_test.disp();
    int good_answer=0;
    for(int i=0;i<X_test.col();i++) {
        int temp_answer=0;
        if(predict.getCoeff(0,i)>=0.5) temp_answer = 1;
        else temp_answer=0;

        if(temp_answer == Y_test.getCoeff(0,i)) good_answer++;
    }
    std::cout << "Number of good answer: " << good_answer <<"/" << Y_test.col() << std::endl;
}