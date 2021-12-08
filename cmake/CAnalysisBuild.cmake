cmake_host_system_information(RESULT HOSTNAME QUERY HOSTNAME)
if (DEFINED IN_GITHUB_ACTIONS)
    set(CTEST_SITE              "Biovault_GHActions")
else()
    set(CTEST_SITE              "${HOSTNAME}")
endif()
set(CTEST_BUILD_NAME        "${CMAKE_HOST_SYSTEM_NAME}-${CMAKE_HOST_SYSTEM_PROCESSOR}-prod")
set(CTEST_SOURCE_DIRECTORY ".")
set(CTEST_BINARY_DIRECTORY ".")
set(CTEST_CMAKE_GENERATOR "Visual Studio 16 2019")
set(CTEST_CONFIGURE_COMMAND "echo Configure in conan")
set(CTEST_BUILD_COMMAND "conan create .. HDILib/1.2.5@lkeb/alpha --profile action_build -s build_type=Release -e Analysis=TRUE")

ctest_start(Experimental GROUP Experimental)  # GROUP was TRACK < 3.16
ctest_build(BUILD "${CTEST_BINARY_DIRECTORY}" RETURN_VALUE build_result)
ctest_submit(HTTPHEADER "Authorization: Bearer ${HDILIB_TOKEN}" RETURN_VALUE build_result)