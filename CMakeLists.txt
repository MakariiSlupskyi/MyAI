cmake_minimum_required(VERSION 3.26)
project(MyAI)

set(CMAKE_CXX_STANDART 14)
set(CMAKE_CXX_STANDART_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

include_directories(
	${PROJECT_SOURCE_DIR}/include
)

# GoogleTest stuff
include(FetchContent)
FetchContent_Declare(
	googletest
	URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_subdirectory(tests)

add_library(myai-functions ${PROJECT_SOURCE_DIR}/src/functions.cpp ${PROJECT_SOURCE_DIR}/include/MyAI/functions.h)
add_library(myai-neural-network ${PROJECT_SOURCE_DIR}/src/NeuralNetwork.cpp ${PROJECT_SOURCE_DIR}/include/MyAI/NeuralNetwork.h)