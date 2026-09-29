cmake_minimum_required(VERSION 2.8.3)
project(aloam_velodyne)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  image_transport
  cv_bridge
  tf
)

#find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

find_package(OpenMP REQUIRED)
find_package(GTSAM REQUIRED QUIET)
find_package(teaserpp REQUIRED)
find_package(TBB REQUIRED)

# Build the vendored small_gicp core only. Keep its optional PCL tools and
# standalone utilities out of the catkin build.
set(BUILD_HELPER ON CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(BUILD_WITH_PCL OFF CACHE BOOL "" FORCE)
set(BUILD_WITH_TBB ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/small_gicp EXCLUDE_FROM_ALL)

# Build the vendored Quatro module directly. Avoid nesting its catkin_package()
# call inside SC-PGO while keeping the documented quattro_module.h API.
add_library(quatro
  third_party/Quatro/src/fpfh.cc
  third_party/Quatro/src/matcher.cc
  third_party/Quatro/src/quatro_module.cc
)
target_include_directories(quatro PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/Quatro/include>
)
include(ProcessorCount)
ProcessorCount(QUATRO_PROC_NUM)
if(NOT QUATRO_PROC_NUM OR QUATRO_PROC_NUM LESS 1)
  set(QUATRO_PROC_NUM 1)
endif()
target_compile_definitions(quatro PRIVATE
  TBB_EN
  TBB_PROC_NUM=${QUATRO_PROC_NUM}
)
target_link_libraries(quatro PUBLIC
  ${PCL_LIBRARIES}
  teaserpp::teaser_registration
  teaserpp::teaser_io
  TBB::tbb
  OpenMP::OpenMP_CXX
)

include_directories(
  include
	${catkin_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL 
  INCLUDE_DIRS include
)

add_executable(alaserPGO 
  src/laserPosegraphOptimization.cpp
  include/scancontext/Scancontext.cpp
)
target_compile_options(alaserPGO 
  PRIVATE ${OpenMP_CXX_FLAGS}
)
target_link_libraries(alaserPGO 
  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
  gtsam
  quatro
  small_gicp
)




