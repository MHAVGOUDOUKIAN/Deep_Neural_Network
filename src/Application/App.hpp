#ifndef APP_HPP
#define APP_HPP

#include <vector>
#include <iostream>

#include <SFML/Graphics.hpp>
#include <Engine/EventHandler.hpp>

class App : public sf::Drawable, public MouseObserver {
    public:
        App();
        virtual ~App();
        void update(sf::Time deltaTime);
        virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;
        virtual void notify(sf::Mouse::Button mouse, sf::Vector2i& pos, bool clicked);
};

#endif