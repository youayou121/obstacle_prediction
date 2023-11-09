cmake_minimum_required(VERSION 3.0.2)
project(obstacle_prediction)
find_package(OpenCV 4.2.0 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  tf2
  pcl_ros
  laser_geometry
)
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES obstacle_prediction
#  CATKIN_DEPENDS opencv4 roscpp rospy tf2
#  DEPENDS system_lib
)
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
add_executable(${PROJECT_NAME} src/obstacle_prediction.cpp)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)
