# Here CURDIR is assumed to be the root folder of the project.

CARLA_ROOT_FOLDER=${CURDIR}
CARLA_BUILD_FOLDER=${CURDIR}/Build
CARLA_DIST_FOLDER=${CURDIR}/Dist
CARLA_BUILD_TOOLS_FOLDER=${CURDIR}/Util/BuildTools
CARLA_TEST_RESULTS_FOLDER=${CARLA_BUILD_FOLDER}/test-results

CARLAUE4_ROOT_FOLDER=${CURDIR}/Unreal/CarlaUE4
CARLAUE4_PLUGIN_ROOT_FOLDER=${CURDIR}/Unreal/CarlaUE4/Plugins/Carla
CARLA_PYTHONAPI_ROOT_FOLDER=${CURDIR}/PythonAPI

LIBCARLA_ROOT_FOLDER=${CURDIR}/LibCarla
LIBCARLA_BUILD_SERVER_FOLDER=${CARLA_BUILD_FOLDER}/libcarla-server-build
LIBCARLA_BUILD_CLIENT_FOLDER=${CARLA_BUILD_FOLDER}/libcarla-client-build
LIBCARLA_INSTALL_SERVER_FOLDER=${CARLAUE4_PLUGIN_ROOT_FOLDER}/CarlaDependencies
LIBCARLA_INSTALL_CLIENT_FOLDER=${CARLA_PYTHONAPI_ROOT_FOLDER}/dependencies

CARLAUE4_PLUGIN_DEPS_FOLDER=${CARLAUE4_PLUGIN_ROOT_FOLDER}/CarlaDependencies

LIBSTDCPP_TOOLCHAIN_FILE=${CARLA_BUILD_FOLDER}/LibStdCppToolChain.cmake
LIBCPP_TOOLCHAIN_FILE=${CARLA_BUILD_FOLDER}/LibCppToolChain.cmake
CMAKE_CONFIG_FILE=${CARLA_BUILD_FOLDER}/CMakeLists.txt.in

LIBCARLA_TEST_CONTENT_FOLDER=${CARLA_BUILD_FOLDER}/test-content
