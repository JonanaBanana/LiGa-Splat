cmake_minimum_required(VERSION 3.8)
project(airlab_lidar_3dgs)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# install scripts
add_library(accumulator_component SHARED src/accumulator.cpp)
ament_target_dependencies(accumulator_component
  rclcpp
  sensor_msgs
  PCL
  pcl_conversions
  rclcpp_components
)
add_library(global_processor_component SHARED src/global_processor.cpp)
ament_target_dependencies(global_processor_component
  rclcpp
  sensor_msgs
  PCL
  pcl_conversions
  rclcpp_components
)

add_library(path_publisher_component SHARED src/path_publisher.cpp)
ament_target_dependencies(path_publisher_component
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  rclcpp_components
)

add_library(image_saver_component SHARED src/image_saver.cpp)
ament_target_dependencies(image_saver_component
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  rclcpp_components
)

add_library(odom_saver_component SHARED src/odom_saver.cpp)
ament_target_dependencies(odom_saver_component
  rclcpp
  nav_msgs
  rclcpp_components
)

rclcpp_components_register_node(
    accumulator_component
    PLUGIN "airlab_lidar_3dgs::PointCloudAccumulator"
    EXECUTABLE accumulator_node
)

rclcpp_components_register_node(
    global_processor_component
    PLUGIN "airlab_lidar_3dgs::GlobalProcessor"
    EXECUTABLE global_processor_node
)

rclcpp_components_register_node(
    path_publisher_component
    PLUGIN "airlab_lidar_3dgs::PathPublisher"
    EXECUTABLE path_publisher_node
)

rclcpp_components_register_node(
    image_saver_component
    PLUGIN "airlab_lidar_3dgs::ImageSaver"
    EXECUTABLE image_saver_node
)

rclcpp_components_register_node(
    odom_saver_component
    PLUGIN "airlab_lidar_3dgs::OdomSaver"
    EXECUTABLE odom_saver_node
)

ament_export_targets(export_accumulator_component)
install(TARGETS accumulator_component
  EXPORT export_accumulator_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  )
ament_export_targets(export_global_processor_component)
install(TARGETS global_processor_component
  EXPORT export_global_processor_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  )

ament_export_targets(export_path_publisher_component)
install(TARGETS path_publisher_component
  EXPORT export_path_publisher_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  )

ament_export_targets(export_image_saver_component)
install(TARGETS image_saver_component
  EXPORT export_image_saver_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  )

  ament_export_targets(export_odom_saver_component)
install(TARGETS odom_saver_component
  EXPORT export_odom_saver_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()



#install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)
#install rviz config
install(
  DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz)


# --- Standalone utilities (no ROS2 dependencies) ---
# --- ROS2 can be used to run the utils (for example: ros2 run airlab_3dgs pcd_viewer)---
add_executable(pcd_viewer utils/pcd_viewer.cpp)
target_include_directories(pcd_viewer PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(pcd_viewer ${PCL_LIBRARIES})

install(TARGETS pcd_viewer
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
