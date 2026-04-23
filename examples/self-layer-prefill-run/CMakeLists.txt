cmake_minimum_required(VERSION 3.10)

project(llama-self-layer-prefill-run)

add_executable(llama-self-layer-prefill-run main.cpp)
target_link_libraries(llama-self-layer-prefill-run PRIVATE llama common)
target_include_directories(llama-self-layer-prefill-run PRIVATE ${CMAKE_SOURCE_DIR}/include)
