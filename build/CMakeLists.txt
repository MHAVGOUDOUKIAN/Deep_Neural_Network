cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME $ENV{TITLE})
project(${PROJECT_NAME})

# Very important, otherwise your executable will be considered as a shared library by your OS and not as an executable
set (CMAKE_CXX_FLAGS "-no-pie")
set (SRC_DIR ${PROJECT_SOURCE_DIR}/../src)

set(PERMISSIONS OWNER_READ 
    # Set the permissions you want for your executable
    # You can choose between EXECUTE/READ/WRITE for OWNER/GROUP/WORLD
    OWNER_EXECUTE
    OWNER_READ
    OWNER_WRITE
    GROUP_EXECUTE 
    WORLD_EXECUTE
)

include_directories(${SRC_DIR})

# Select the sources and headers files to use in the current directory
file(GLOB_RECURSE SRCS ../src/*.cpp)
file(GLOB_RECURSE HDRS ../src/*.hpp)

add_executable(${PROJECT_NAME} # Name of the executable
    ${SRCS}
    ${HDRS})
    
target_link_libraries(${PROJECT_NAME}
    sfml-graphics
    sfml-window
    sfml-system 
    sfml-audio
    sfml-network
    pthread
)

install(FILES ${PROJECT_NAME}  DESTINATION ${PROJECT_SOURCE_DIR}/.. PERMISSIONS ${PERMISSIONS} )
