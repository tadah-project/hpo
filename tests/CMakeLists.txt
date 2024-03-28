
file(GLOB tests_src "*.cpp")
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tests_data DESTINATION .)

add_executable(test_md ${tests_src})
target_link_libraries(test_md PRIVATE 
    Catch2 tadah.md tadah.core tadah.models) 

include(Catch.cmake)
catch_discover_tests(test_md)
