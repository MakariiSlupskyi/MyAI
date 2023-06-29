add_executable(my_test
    test.cpp
)
target_link_libraries(my_test
    myai-neural-network
    myai-functions
    GTest::gtest_main
)

include (GoogleTest)
gtest_discover_tests(my_test)