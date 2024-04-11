#include <Application/App.hpp>

App::App() {
    EventHandler::getEventHandler()->addMouseObserver(this);
}

App::~App() {}

void App::update(sf::Time deltaTime) {

}

void App::notify(sf::Mouse::Button mouse, sf::Vector2i& pos, bool clicked) {
}

void App::draw(sf::RenderTarget& target, sf::RenderStates states) const {
}