add_executable(regression regression.c)
target_include_directories(regression PRIVATE ../include)
target_link_libraries(regression m)
