cmake_minimum_required(VERSION 3.10)

project(llama-spec-prefill-run)

add_executable(llama-spec-prefill-run main.cpp)
target_link_libraries(llama-spec-prefill-run PRIVATE llama common)
target_include_directories(llama-spec-prefill-run PRIVATE ${CMAKE_SOURCE_DIR}/include)
