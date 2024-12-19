cmake_minimum_required(VERSION 3.10)

project(GestureAccess LANGUAGES CXX)

find_package(OpenCV REQUIRED)

add_executable(gesture_access gesture_access.cpp)

target_link_libraries(gesture_access PRIVATE ${OpenCV_LIBS})

set_target_properties(gesture_access PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

