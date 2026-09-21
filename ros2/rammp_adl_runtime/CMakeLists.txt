cmake_minimum_required(VERSION 3.16)
project(rammp_adl_runtime VERSION 0.1.0)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Both pip and ROS install this one implementation. rosidl generation belongs to
# the separate rammp_adl_interfaces package, as required by ament_cmake_python.
get_filename_component(RAMMP_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
ament_python_install_package(rammp_adl
  PACKAGE_DIR "${RAMMP_PROJECT_ROOT}/rammp_adl"
  VERSION "${PROJECT_VERSION}"
)

install(PROGRAMS scripts/rammp_adl_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)
install(DIRECTORY
  "${RAMMP_PROJECT_ROOT}/skills"
  "${RAMMP_PROJECT_ROOT}/schemas"
  "${RAMMP_PROJECT_ROOT}/config"
  "${RAMMP_PROJECT_ROOT}/examples"
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)
install(FILES "${RAMMP_PROJECT_ROOT}/tools/check_design.py"
  DESTINATION share/${PROJECT_NAME}/tools
)
# Vendor asset licenses and source provenance travel with the optional physics
# fixtures; generated screenshots/reports are not deployment assets.
if(EXISTS "${RAMMP_PROJECT_ROOT}/simulation/vendor")
  install(DIRECTORY "${RAMMP_PROJECT_ROOT}/simulation/vendor"
    DESTINATION share/${PROJECT_NAME}/simulation
  )
endif()
install(FILES "${RAMMP_PROJECT_ROOT}/simulation/rolling_scene.xml"
  DESTINATION share/${PROJECT_NAME}/simulation
)

ament_package()
