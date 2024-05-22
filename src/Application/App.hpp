#ifndef APP_HPP
#define APP_HPP

#include <vector>
#include <iostream>
#include <iomanip>

#include <SFML/Graphics.hpp>
#include <Engine/EventHandler.hpp>
#include <Application/Matrix.hpp>
#include <Application/NeuralNetwork.hpp>
#include <exception>

class App : public sf::Drawable, public KeyBoardObserver {
    public:
        App();
        virtual ~App();
        void update(sf::Time deltaTime);
        virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;
        virtual void notify(sf::Keyboard::Key key, bool pressed);

    private:
        void generate_data_linear(Matrix& X_feature,Matrix& Y_class, bool update_graphics=false);
        void generate_data_circle(Matrix& X_feature,Matrix& Y_class, bool update_graphics=false);
        void generate_data_3_class(Matrix& X_feature,Matrix& Y_class, bool update_graphics=false);
        void train_simpleNeuron(const Matrix& X_train,const Matrix& Y_train, const int epoch=10, const float learning_rate=1.0f, const bool show_result=true);
        void train_doubleLayer(const Matrix& X_train,const Matrix& Y_train, const int epoch=10, const float learning_rate=1.0f, const bool show_result=true);
        void predict(const Matrix& X_test,const Matrix& Y_test);
    
    private:
        sf::VertexArray m_data_coord; 
        sf::VertexArray m_frontier;

        std::vector<Matrix> m_weights;
        std::vector<Matrix> m_bias;

        NeuralNetwork* nn;
};

#endif