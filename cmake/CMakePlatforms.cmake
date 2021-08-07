cmake_minimum_required(VERSION 3.5)

# This module is shared; use include blocker.
if( _PLATFORMS_ )
	return()
endif()

# Mark it as processed
set(_PLATFORMS_ 1)

include(CMakeCompiler)

# Detect target platform
if( ${CMAKE_SYSTEM_NAME} STREQUAL "Windows" )
	set(PLATFORM_WINDOWS 1)
	set(PLATFORM_NAME "windows")
	add_definitions(-DWINDOWSPC)
elseif( ${CMAKE_SYSTEM_NAME} STREQUAL "Linux" )
	set(PLATFORM_LINUX 1)
	set(PLATFORM_NAME "linux")
	add_definitions(-DLINUXPC)
endif()

message(STATUS "Detected platform: ${PLATFORM_NAME}")

# Detect target architecture
if(PLATFORM_WINDOWS AND CMAKE_CL_64)
	set(PLATFORM_64BIT 1)
endif()

# Configure CMake global variables
set(CMAKE_INSTALL_MESSAGE LAZY)
# Set the output folders based on the identifier
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIRECTORY}/lib/${PROJECT_PLATFORM_NAME})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIRECTORY}/lib/${PROJECT_PLATFORM_NAME})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIRECTORY}/bin/${PROJECT_PLATFORM_NAME})

if( PLATFORM_WINDOWS )
	set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD 1)
endif()

# Find opencl and enable it if possible
FIND_PACKAGE(OpenCL)
if (OpenCL_FOUND)
	add_definitions(-DOPENCL_SUPPORTED)
endif()
