cmake_minimum_required(VERSION 3.0.2)
project(obstacle_prediction)
set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  tf2
  pcl_ros
  laser_geometry
  message_generation
)
add_message_files(
  FILES
  Obstacle.msg
  ObstacleArray.msg
)
generate_messages(
  DEPENDENCIES 
  std_msgs
  geometry_msgs
)
catkin_package(
    CATKIN_DEPENDS message_generation message_runtime std_msgs roscpp rospy 
)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  include
)
add_executable(${PROJECT_NAME} src/obstacle_prediction.cpp)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
