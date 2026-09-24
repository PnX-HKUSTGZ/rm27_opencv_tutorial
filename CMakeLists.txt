cmake_minimum_required(VERSION 3.16)
project(rm27_armor_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio)

add_executable(armor_demo src/main.cpp)
target_include_directories(armor_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(armor_demo PRIVATE ${OpenCV_LIBS})
