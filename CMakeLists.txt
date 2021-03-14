cmake_minimum_required(VERSION 3.5)
project(conc_benchmark_cuda)

# ------------------------------------------------------------------------------
# Setup project.

# Check if CUDA is available.
include(CheckLanguage)
check_language(CUDA)

# Check if a CUDA compiler exists.
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  message(STATUS "Cuda compiler has been found: ${CMAKE_CUDA_COMPILER}")
else(CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "Cuda compiler has not been found")
endif()

# Check host compiler.
if(DEFINED ${CMAKE_CUDA_HOST_COMPILER})
    message(STATUS "Cuda host compiler defined: ${CMAKE_CUDA_HOST_COMPILER}")
endif()

# Check correct compiler id.
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    message(STATUS "Cuda compiler id match NVIDIA")
else(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    message(FATAL_ERROR "Cuda compiler id doesn't match")
endif()

message(STATUS 
        "CMake setup project succeeds with NVIDIA "
        "v${CMAKE_CUDA_COMPILER_VERSION}")

# ------------------------------------------------------------------------------
# Set the C++ compiler options.
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-stack-protector")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=i686")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic-errors")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-align")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-qual")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdisabled-optimization")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-equal")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=2")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat-nonliteral")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat-security")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat-y2k")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wimport")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winit-self")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winline")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winvalid-pch")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-long-long")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-field-initializers")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-format-attribute")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-include-dirs")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpacked")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpointer-arith")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wredundant-decls")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wstack-protector")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wstrict-aliasing=2")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wswitch")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wswitch-default")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wswitch-enum")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunreachable-code")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-function")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-label")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-parameter")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-value")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-variable")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wvariadic-macros")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wwrite-strings")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsign-compare")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsign-conversion")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmessage-length=0")

# ------------------------------------------------------------------------------
# Set the CUDA compiler options.
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_50,code=sm_50")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_52,code=sm_52")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_61,code=sm_61")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=compute_75")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_62")

# ------------------------------------------------------------------------------
# Add the includes.
include_directories(
  include/conc_benchmark_cuda
)

# ------------------------------------------------------------------------------
# Find dependencies.
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

set(DEPENDENCIES 
  rclcpp
  std_msgs
)

# ------------------------------------------------------------------------------
# Library CUDA.
set(CUDA_LIB conc_bench_cuda_lib)
set(CUDA_LIB_SRC 
  src/conc_bench_utils.cu
)

add_library(${CUDA_LIB} STATIC ${CUDA_LIB_SRC})
# add_library(${CUDA_LIB} SHARED ${CUDA_LIB_SRC})
# set_target_properties(${CUDA_LIB} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ------------------------------------------------------------------------------
# Add interfaces.
set(PROJECT_INTERFACES conc_benchmark_interfaces)

set(MSG_FILES
  "msg/ConcArray.msg"
)

set(SRV_FILES
  
)

rosidl_generate_interfaces(${PROJECT_NAME} 
  ${MSG_FILES}
  ${SRV_FILES}
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# ------------------------------------------------------------------------------
# NODE: matrix_supplier
set(NODE_MATRIX_SUPPLIER matrix_supplier)
set(NODE_MATRIX_SUPPLIER_SRC
  src/matrix_supplier.cpp
)

add_executable(${NODE_MATRIX_SUPPLIER} ${NODE_MATRIX_SUPPLIER_SRC})
ament_target_dependencies(${NODE_MATRIX_SUPPLIER} 
  ${DEPENDENCIES}
)

#target_include_directories(${NODE_MATRIX_SUPPLIER} PUBLIC ${CUDA_LIB})
target_link_libraries(${NODE_MATRIX_SUPPLIER} ${CUDA_LIB})

rosidl_target_interfaces(${NODE_MATRIX_SUPPLIER}
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

# ------------------------------------------------------------------------------
# NODE: gpu_exec
set(NODE_GPU_EXEC gpu_exec)
set(NODE_GPU_EXEC_SRC
  src/gpu_exec.cpp
)

add_executable(${NODE_GPU_EXEC} ${NODE_GPU_EXEC_SRC})
ament_target_dependencies(${NODE_GPU_EXEC} ${DEPENDENCIES})

#target_include_directories(${NODE_GPU_EXEC} PUBLIC ${CUDA_LIB})
target_link_libraries(${NODE_GPU_EXEC} ${CUDA_LIB})

rosidl_target_interfaces(${NODE_GPU_EXEC}
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

# ------------------------------------------------------------------------------
# NODE: cpu_exec
set(NODE_CPU_EXEC cpu_exec)
set(NODE_CPU_EXEC_SRC
  src/cpu_exec.cpp
)

add_executable(${NODE_CPU_EXEC} ${NODE_CPU_EXEC_SRC})
ament_target_dependencies(${NODE_CPU_EXEC} ${DEPENDENCIES})

#target_include_directories(${NODE_CPU_EXEC} PUBLIC ${CUDA_LIB})
target_link_libraries(${NODE_CPU_EXEC} ${CUDA_LIB})

rosidl_target_interfaces(${NODE_CPU_EXEC}
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

# ------------------------------------------------------------------------------
# Install directives.

install(TARGETS
  ${NODE_MATRIX_SUPPLIER}
  ${NODE_GPU_EXEC}
  ${NODE_CPU_EXEC}
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

# ------------------------------------------------------------------------------
# Testing directives.

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
