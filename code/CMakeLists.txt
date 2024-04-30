cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(ImplicitFeatureEncoding)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find packages
find_package(PCL REQUIRED)

find_package(OpenMP)
if (OpenMP_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
    link_libraries(${OpenMP_CXX_FLAGS})
endif ()

# Set project sources
set(PROJECT_SOURCES
        implicit_feature_encoding/implicit_feature_encoding.cpp
        main.cpp
)

# Add executable target
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# Add include directories to target
target_include_directories(${PROJECT_NAME} PUBLIC
        ${PCL_INCLUDE_DIRS}
        3rd_party
        implicit_feature_encoding
)

# Link libraries to target
target_link_libraries(${PROJECT_NAME} PUBLIC
        ${PCL_LIBRARIES}
)