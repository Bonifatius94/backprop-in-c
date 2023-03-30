add_executable(MatrixTests matrix_tests.c)
target_include_directories(MatrixTests PRIVATE ../include)
target_link_libraries(MatrixTests m)

add_executable(ModelTests model_tests.c)
target_include_directories(ModelTests PRIVATE ../include)
target_link_libraries(ModelTests m)
