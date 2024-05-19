#include <Application/App.hpp>

#define log(str) std::cout << str << std::endl;

App::App() {
    EventHandler::getEventHandler()->addMouseObserver(this);
    m_data_coord = sf::VertexArray(sf::Quads,4);
    m_frontier = sf::VertexArray(sf::Lines, 2);
    m_bias.clear();
    m_weights.clear();

    // Generate training data
    int data_number_train{1000};
    Matrix X_train{Matrix(2,data_number_train)}, Y_train{Matrix(1,data_number_train)};
    generate_data_linear(X_train, Y_train,true);

    // Trainings
    train_simpleNeuron(X_train, Y_train,1000);

    // Generate testing data
    int data_number_test{10};
    Matrix X_test{Matrix(2,data_number_test)}, Y_test{Matrix(1,data_number_test)};
    generate_data_linear(X_test, Y_test, true);
    //X_test.disp();
    //Y_test.disp();

    // Predictions
    predict(X_test, Y_test);
    
}

App::~App() {}

void App::update(sf::Time deltaTime) {

}

void App::notify(sf::Mouse::Button mouse, sf::Vector2i& pos, bool clicked) {
}

void App::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    target.draw(m_data_coord);
    target.draw(m_frontier);
}

// This function generate linearly separable data
// X_feature max dim : row=2, col= nb_of_data
// Y_class max dim: row=1, col=nb_of_data
void App::generate_data_linear(Matrix& X_feature, Matrix& Y_class, bool update_graphics) {    
    const float a{-1.4f}, b{1.0f}, c{0.3f};
    const float zoom{500.f};
    sf::Color class_color{sf::Color::Cyan};
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
}

void App::train_simpleNeuron(const Matrix& X_train,const Matrix& Y_train, const int epoch, const float learning_rate) {
    int features = X_train.col();
    m_weights.push_back(Matrix(1,X_train.row()));
    m_bias.push_back(Matrix(1,1));
    for(int iter=0; iter<epoch; iter++) {        
        // Forward propagation
        Matrix Z {m_weights[0]*X_train};
        Z+m_bias[0].getCoeff(0,0);
        Matrix A{Z};
        A.applySigmo();

        // Calc lost function
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
        
        // Back propagation
        Matrix dW = Y_train;
        dW*(-1);
        dW = dW+A;
        dW = X_train*dW.transposee();
        dW*(1.f/(float)res.col());

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

        m_bias[0].setCoeff(0,0, m_bias[0].getCoeff(0,0) - learning_rate * db_L);
    }
}

void App::train_doubleLayer(const Matrix& X_train,const Matrix& Y_train, const int epoch, const float learning_rate) {

}

void App::predict(const Matrix& X_test,const Matrix& Y_test) {
    log("");
    log("");
    log("PREDICTIONS")
    Matrix Z {m_weights[0]*X_test};
    Z+m_bias[0].getCoeff(0,0);
    Matrix A{Z};
    A.applySigmo();
    X_test.disp();
    A.disp();
    Y_test.disp();
}