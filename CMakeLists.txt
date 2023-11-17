cmake_minimum_required(VERSION 3.0)

project(tp2)

# Define your source files
file(GLOB SOURCES "src/*.cpp")

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Create an executable target
add_executable(tp2 ${SOURCES})

# Link the OpenCV libraries to your executable
target_link_libraries(tp2 ${OpenCV_LIBS})