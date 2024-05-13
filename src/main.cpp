#include <iostream>
#include <Engine/Engine.hpp>
#include <Tests/Matrix_test.hpp>

int main() {

    #ifdef ENABLE_TEST
        Engine e;
        e.run();
    #else
        launch_test();
    #endif

    return 0;
}
