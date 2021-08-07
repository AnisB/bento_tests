cmake_minimum_required(VERSION 3.2)

bento_exe("allocator_tester" "tests" "allocator_tester.cpp" "${BENTO_SDK_INCLUDE};${BENTO_TESTS_3RD_INCLUDE}")
target_link_libraries("allocator_tester" "bento_sdk")

bento_exe("string_tester" "tests" "string_tester.cpp" "${BENTO_SDK_INCLUDE};${BENTO_TESTS_3RD_INCLUDE}")
target_link_libraries("string_tester" "bento_sdk")

if (OpenCL_FOUND)
	bento_exe("compute_tester" "tests" "compute_tester.cpp" "${BENTO_SDK_INCLUDE};${BENTO_TESTS_3RD_INCLUDE}")
	target_link_libraries("compute_tester" "bento_sdk" "${OpenCL_LIBRARIES}")
endif()