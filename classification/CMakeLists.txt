add_executable(classification classification.c)
target_include_directories(classification PRIVATE ../include)
target_include_directories(classification PRIVATE ../mnist_utils)
target_link_libraries(classification m)
