// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#ifndef LIBCARLA_SENSOR_REGISTRY_INCLUDE_H
#define LIBCARLA_SENSOR_REGISTRY_INCLUDE_H

#include "carla/sensor/CompositeSerializer.h"

// =============================================================================
// Follow the 4 steps to register a new sensor.
// =============================================================================

// 1. Include the serializer here.
#include "carla/sensor/s11n/ImageSerializer.h"
#include "carla/sensor/s11n/LidarSerializer.h"

// 2. Add a forward-declaration of the sensor here.
class ADepthCamera;
class ARayCastLidar;
class ASceneCaptureCamera;
class ASemanticSegmentationCamera;

namespace carla {
namespace sensor {

  // 3. Register the sensor and its serializer in the SensorRegistry.

  /// Contains a registry of all the sensors available and allows serializing
  /// and deserializing sensor data for the types registered.
  using SensorRegistry = CompositeSerializer<
    std::pair<ASceneCaptureCamera *, s11n::ImageSerializer>,
    std::pair<ADepthCamera *, s11n::ImageSerializer>,
    std::pair<ASemanticSegmentationCamera *, s11n::ImageSerializer>,
    std::pair<ARayCastLidar *, s11n::LidarSerializer>
  >;

} // namespace sensor
} // namespace carla

#endif // LIBCARLA_SENSOR_REGISTRY_INCLUDE_H

#ifdef LIBCARLA_SENSOR_REGISTRY_WITH_SENSOR_INCLUDES

// 4. Include the sensor here.
#include "Carla/Sensor/DepthCamera.h"
#include "Carla/Sensor/RayCastLidar.h"
#include "Carla/Sensor/SceneCaptureCamera.h"
#include "Carla/Sensor/SemanticSegmentationCamera.h"

#endif // LIBCARLA_SENSOR_REGISTRY_WITH_SENSOR_INCLUDES
