cmake_minimum_required(VERSION 2.8.3)
project(get_camera)
find_package(catkin REQUIRED COMPONENTS cv_bridge image_transport roscpp std_msgs sensor_msgs message_generation)

find_package(OpenCV REQUIRED)

catkin_package(
  
)


include_directories(
  include ${catkin_INCLUDE_DIRS}
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  src/
  include/
)




message(STATUS "opencv version: ${OpenCV_VERSION}")
message(STATUS "opencv include: ${OpenCV_INCLUDE_DIRS}")