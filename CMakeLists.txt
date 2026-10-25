cmake_minimum_required(VERSION 3.20)

project(adaptive VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ADAPTIVE_NATIVE_OPTIMIZATIONS "Enable -march=native optimizations for the library target" OFF)

add_library(adaptive INTERFACE)

target_compile_features(adaptive INTERFACE cxx_std_20)

target_compile_definitions(adaptive
  INTERFACE
    $<$<CONFIG:Release>:NDEBUG>
)

if(ADAPTIVE_NATIVE_OPTIMIZATIONS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(adaptive
    INTERFACE
      $<$<CONFIG:Release>:-march=native>
  )
endif()

target_include_directories(adaptive
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_sources(adaptive
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/adaptive/adaptive.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/adaptive/assert.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/adaptive/bit_util.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/adaptive/policy.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/adaptive/tagged_node_ptr.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/adaptive/array/array_node_ptr.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/adaptive/array/array_vector.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/adaptive/array/dynamic_array_node.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/adaptive/array/leaf_array_node.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/adaptive/array/static_array_node.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/adaptive/bit_vector/bit_vector.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/adaptive/bit_vector/bv_node_ptr.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/adaptive/bit_vector/dynamic_bv_node.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/adaptive/bit_vector/leaf_bv_node.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/adaptive/bit_vector/static_bv_node.hpp>
)

add_library(adaptive::adaptive ALIAS adaptive)

# Installation and packaging
install(TARGETS adaptive
  EXPORT adaptiveTargets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT adaptiveTargets
  FILE adaptiveTargets.cmake
  NAMESPACE adaptive::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/adaptive
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/adaptiveConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/adaptiveConfig.cmake.in"
"@PACKAGE_INIT@

include(\"\${CMAKE_CURRENT_LIST_DIR}/adaptiveTargets.cmake\")

check_required_components(adaptive)
"
)

configure_package_config_file(
  "${CMAKE_CURRENT_BINARY_DIR}/adaptiveConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/adaptiveConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/adaptive
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/adaptiveConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/adaptiveConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/adaptive
)

# Benchmarks
option(ADAPTIVE_BUILD_BENCHMARKS "Build adaptive benchmark binaries" OFF)

if(ADAPTIVE_BUILD_BENCHMARKS)
  foreach(bench_name IN ITEMS access rank select)
    add_executable(adaptive_bench_${bench_name}
      benchmarks/adaptive/${bench_name}.cpp
    )
    target_link_libraries(adaptive_bench_${bench_name}
      PRIVATE
        adaptive::adaptive
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
      target_compile_options(adaptive_bench_${bench_name}
        PRIVATE
          $<$<CONFIG:Release>:-O3>
          $<$<CONFIG:Release>:-march=native>
      )
    endif()
    set_target_properties(adaptive_bench_${bench_name}
      PROPERTIES
        OUTPUT_NAME ${bench_name}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bench/adaptive
    )
  endforeach()
endif()
