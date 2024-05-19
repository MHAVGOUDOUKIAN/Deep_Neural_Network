#ifndef APP_HPP
#define APP_HPP

#include <vector>
#include <iostream>

#include <SFML/Graphics.hpp>
#include <Engine/EventHandler.hpp>
#include <Application/Matrix.hpp>
#include <exception>

class App : public sf::Drawable, public MouseObserver {
    public:
        App();
        virtual ~App();
        void update(sf::Time deltaTime);
        virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;
        virtual void notify(sf::Mouse::Button mouse, sf::Vector2i& pos, bool clicked);

    private:
        void generate_data_linear(Matrix& X_feature,Matrix& Y_class, bool update_graphics=false);
        void train_simpleNeuron(const Matrix& X_train,const Matrix& Y_train, const int epoch=10, const float learning_rate=1.0f);
        void predict(const Matrix& X_test,const Matrix& Y_test);
    
    private:
        sf::VertexArray m_data_coord; 
        sf::VertexArray m_frontier;

        Matrix m_weight;
        float m_bias;
};

#endif