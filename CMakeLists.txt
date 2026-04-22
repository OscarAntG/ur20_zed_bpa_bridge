cmake_minimum_required(VERSION 3.8)
project(ur20_zed_bpa_bridge)
cmake_policy(SET CMP0074 NEW)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io features search filters segmentation sample_consensus)

add_subdirectory(third_party_git/bpa)

add_executable(bpa_bridge src/bpa_bridge.cpp)
add_executable(noether_surface_fitting src/noether_surface_fitting.cpp)

target_include_directories(bpa_bridge PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_include_directories(noether_surface_fitting PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)

ament_target_dependencies(bpa_bridge
  rclcpp
  sensor_msgs
  std_srvs
  moveit_msgs
  shape_msgs
  visualization_msgs
  pcl_conversions
)

ament_target_dependencies(noether_surface_fitting
  rclcpp
  sensor_msgs
  std_srvs
  shape_msgs
  visualization_msgs
  geometry_msgs
  tf2
  tf2_ros
  pcl_conversions
)

target_link_libraries(bpa_bridge ${PCL_LIBRARIES} libbpa)
target_link_libraries(noether_surface_fitting ${PCL_LIBRARIES})

install(TARGETS 
  bpa_bridge 
  noether_surface_fitting
  DESTINATION lib/${PROJECT_NAME})

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

ament_package()
