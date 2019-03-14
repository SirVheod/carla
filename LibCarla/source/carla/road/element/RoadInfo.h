// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/road/element/RoadInfoVisitor.h"
#include "carla/NonCopyable.h"

#include <map>
#include <string>
#include <vector>

namespace carla {
namespace road {
namespace element {

  class RoadInfo : private NonCopyable {
  public:

    virtual ~RoadInfo() = default;

    virtual void AcceptVisitor(RoadInfoVisitor &) = 0;

    float GetDistance() const {
      return s;
    }

    // distance from road's start location
    float s;

  protected:

    RoadInfo(float distance = 0) : s(distance) {}
  };

} // namespace element
} // namespace road
} // namespace carla
