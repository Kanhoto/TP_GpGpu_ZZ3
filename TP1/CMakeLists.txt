cmake_minimum_required (VERSION 3.16)
project(ISIMA_TP_1)

include("FetchContent")
FetchContent_Declare(
	stb
	GIT_REPOSITORY https://github.com/nothings/stb.git
)
FetchContent_GetProperties(stb)
if(NOT stb)
	FetchContent_Populate(stb)
	#include the headerfiles and libs
endif()

add_executable(ISIMA_TP_1 src/main.cpp)

set_property(TARGET ISIMA_TP_1 PROPERTY CXX_STANDARD 11)
target_include_directories(ISIMA_TP_1 PRIVATE ${stb_SOURCE_DIR})
