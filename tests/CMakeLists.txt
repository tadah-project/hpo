
file(GLOB tests_src "*.cpp")
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tests_data DESTINATION .)

add_executable(test_hpo ${tests_src})
target_link_libraries(test_hpo PRIVATE 
    Catch2 tadah.core tadah.mlip tadah.hpo) 

include(Catch.cmake)
catch_discover_tests(test_hpo)
