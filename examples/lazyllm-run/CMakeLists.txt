cmake_minimum_required(VERSION 3.10)
project(llama-lazyllm-run)

add_executable(llama-lazyllm-run main.cpp)
target_link_libraries(llama-lazyllm-run PRIVATE llama common)
target_include_directories(llama-lazyllm-run PRIVATE ${CMAKE_SOURCE_DIR}/include)
