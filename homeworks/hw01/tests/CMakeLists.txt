add_executable(hw01_tests
  test_array.cpp
  test_engine.cpp
)
target_link_libraries(hw01_tests PRIVATE hw01_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(hw01_tests)
