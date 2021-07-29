cmake_minimum_required(VERSION 3.5)

macro(define_plaform_settings)
	if( PLATFORM_WINDOWS)
		add_compile_options(/Zi)
		add_compile_options($<$<CONFIG:DEBUG>:/Od> $<$<NOT:$<CONFIG:DEBUG>>:/Ox>)
		add_compile_options(/Ob2)
		add_compile_options($<$<NOT:$<CONFIG:DEBUG>>:/Oi>)
		add_compile_options(/Ot)
		add_compile_options($<$<NOT:$<CONFIG:DEBUG>>:/GT>)
		add_compile_options(/GF)
		add_compile_options(/GR)

		if( PLATFORM_WINDOWS AND RUNTIME_TYPE STREQUAL "mt")
			add_compile_options($<$<CONFIG:DEBUG>:/MTd> $<$<NOT:$<CONFIG:DEBUG>>:/MT>)
		elseif( PLATFORM_WINDOWS AND RUNTIME_TYPE STREQUAL "md")
			add_compile_options($<$<CONFIG:DEBUG>:/MDd> $<$<NOT:$<CONFIG:DEBUG>>:/MD>)
		endif()
		add_compile_options(/Gy)
		add_compile_options(/fp:fast)
		replace_compile_flags("/GR" "/GR-")
		add_compile_options(/W4)
		add_compile_options(/WX)
		add_exe_linker_flags(/DEBUG)
		add_exe_linker_flags(/MAP)
		replace_linker_flags("/INCREMENTAL" "/INCREMENTAL:NO" debug)
		add_compile_options(/MP)
		add_compile_options(-D_HAS_EXCEPTIONS=0)
		replace_linker_flags("/debug" "/DEBUG" debug)
		replace_linker_flags("/machine:x64" "/MACHINE:X64")
		add_compile_options(-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
		add_compile_options(-DSECURITY_WIN32)
	elseif(PLATFORM_LINUX)
		# Debug information
		if( ENGINE_USE_DEBUG_INFO )
			add_compile_options(-g)
			add_exe_linker_flags("-rdynamic")
		else()
			add_compile_options($<$<CONFIG:DEBUG>:-g>)
			add_exe_linker_flags("-rdynamic" debug)
		endif()

		# Enable full optimization in dev/release
		add_compile_options($<$<CONFIG:DEBUG>:-O0> $<$<NOT:$<CONFIG:DEBUG>>:-O3>)

		# Use pipes rather than temporary files for communication between the various stages of compilation
		add_compile_options(-pipe)

		# Adds support for multithreading with the pthreads library
		add_compile_options(-pthread)

		# Use fast floating point model
		add_compile_options(-ffast-math)

		# Disable run-time type information (RTTI)
		add_compile_options(-fno-rtti)

		# Enable SIMD instructions (SSE3)
		add_compile_options(-msse3)

		# Disable specific warnings
		add_compile_options(-Wno-parentheses -Wno-reorder -Wno-missing-braces -Wno-unused-private-field -Wno-return-type-c-linkage -Wno-narrowing)

		# Treat all other warnings as errors
		add_compile_options(-Werror)
		
		set(CMAKE_POSITION_INDEPENDENT_CODE ON)

		# Enable C++11 language, but only for c++ files
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	else()
		message(FATAL_ERROR "Unknown platform!")
	endif()
endmacro()
