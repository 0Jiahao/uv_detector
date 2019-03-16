# set project's name
PROJECT( uvtest )

###############################################################################
# CMake settings
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
set (CMAKE_CXX_STANDARD 11)

SET(CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/ )

# OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Eigen3 3.3 REQUIRED NO_MODULE)
INCLUDE_DIRECTORIES(include)

FILE(GLOB_RECURSE HDRS_FILES "src/*.h" "src/*.hpp" "include/*.h" "include/*.hpp")
FILE(GLOB_RECURSE SRCS_FILES "src/*.c" "src/*.cpp" "include/*.c" "include/*.cpp")

ADD_EXECUTABLE(${PROJECT_NAME} ${SRCS_FILES} ${HDRS_FILES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS} Eigen3::Eigen)

