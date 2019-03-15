// Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy; see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/NonCopyable.h"
#include <string>
#include <vector>
#include "carla/road/RoadTypes.h"
#include "carla/road/general/Validity.h"

namespace carla {
namespace road {
namespace signal {

  class SignalDependency : private MovableNonCopyable {
  public:

    SignalDependency(
        int32_t road_id,
        uint32_t signal_id,
        uint32_t dependency_id,
        std::string type)
        : _road_id(road_id),
          _signal_id(signal_id),
          _dependency_id(dependency_id),
          _type(type) {}

  private:

#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wunused-private-field"
#endif
    int32_t _road_id;
    int32_t _signal_id;
    int32_t _dependency_id;
    std::string _type;
#if defined(__clang__)
#  pragma clang diagnostic pop
#endif
  };



} // object
} // road
} // carla
