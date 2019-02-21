// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include <carla/rpc/VehicleControl.h>
#include <carla/rpc/VehiclePhysicsControl.h>
#include <carla/rpc/WheelPhysicsControl.h>
#include <carla/rpc/WalkerControl.h>

#include <ostream>

namespace carla {
namespace rpc {

  static auto boolalpha(bool b) {
    return b ? "True" : "False";
  };

  std::ostream &operator<<(std::ostream &out, const VehicleControl &control) {
    out << "VehicleControl(throttle=" << control.throttle
        << ", steer=" << control.steer
        << ", brake=" << control.brake
        << ", hand_brake=" << boolalpha(control.hand_brake)
        << ", reverse=" << boolalpha(control.reverse)
        << ", manual_gear_shift=" << boolalpha(control.manual_gear_shift)
        << ", gear=" << control.gear << ')';
    return out;
  }

  std::ostream &operator<<(std::ostream &out, const WalkerControl &control) {
    out << "WalkerControl(direction=" << control.direction
        << ", speed=" << control.speed
        << ", jump=" << boolalpha(control.jump) << ')';
    return out;
  }

  std::ostream &operator<<(std::ostream &out, const WheelPhysicsControl &control) {
    out << "WheelPhysicsControl(tire_friction=" << control.tire_friction 
        << ", damping_rate=" << control.damping_rate
        << ", steer_angle=" << control.steer_angle
        << ", disable_steering=" << boolalpha(control.disable_steering) << ')';
    return out;
  }

  std::ostream &operator<<(std::ostream &out, const VehiclePhysicsControl &control) {
    out << "VehiclePhysicsControl(torque_curve=" << control.torque_curve
    << ", max_rpm=" << control.max_rpm
    << ", moi=" << control.moi
    << ", damping_rate_full_throttle=" << control.damping_rate_full_throttle
    << ", damping_rate_zero_throttle_clutch_engaged=" << control.damping_rate_zero_throttle_clutch_engaged
    << ", damping_rate_zero_throttle_clutch_disengaged=" << control.damping_rate_zero_throttle_clutch_disengaged 
    << ", use_gear_autobox=" << boolalpha(control.use_gear_autobox)
    << ", gear_switch_time=" << control.gear_switch_time
    << ", clutch_strength=" << control.clutch_strength
    << ", mass=" << control.mass
    << ", drag_coefficient=" << control.drag_coefficient
    << ", inertia_tensor_scale=" << control.inertia_tensor_scale 
    << ", center_of_mass=" << control.center_of_mass 
    << ", steering_curve=" << control.steering_curve 
    << ", wheels=" << control.wheels << ')';
    return out;
  }
} // namespace rpc
} // namespace carla



static auto GetWheels(const carla::rpc::VehiclePhysicsControl &self) {
  return boost::python::list(self.GetWheels());
}

static void SetWheels(carla::rpc::VehiclePhysicsControl &self, const boost::python::list &list) {  
  std::vector<carla::rpc::WheelPhysicsControl> wheels;
  auto length = boost::python::len(list);
  for (auto i = 0u; i < length; ++i) {
    wheels.push_back(boost::python::extract<carla::rpc::WheelPhysicsControl &>(list[i]));
  }
  self.wheels = wheels;
}

static auto GetTorqueCurve(const carla::rpc::VehiclePhysicsControl &self) {
  return boost::python::list(self.GetTorqueCurve());
}

static void SetTorqueCurve(carla::rpc::VehiclePhysicsControl &self, const boost::python::list &list) {  
  std::vector<carla::geom::Vector2D> torque_curve;
  auto length = boost::python::len(list);
  torque_curve.reserve(length);
  for (auto i = 0u; i < length; ++i) {
    boost::python::extract<carla::geom::Vector2D> ext(list[i]);
    if (ext.check())
      torque_curve.push_back(ext);
    else {
      torque_curve.push_back(carla::geom::Vector2D{
          boost::python::extract<float>(list[i][0u]),
          boost::python::extract<float>(list[i][1u])});
    }
  }
  self.torque_curve = torque_curve;
}

static auto GetSteeringCurve(const carla::rpc::VehiclePhysicsControl &self) {
  return boost::python::list(self.GetSteeringCurve());
}

static void SetSteeringCurve(carla::rpc::VehiclePhysicsControl &self, const boost::python::list &list) {  
  std::vector<carla::geom::Vector2D> steering_curve;
  auto length = boost::python::len(list);
  for (auto i = 0u; i < length; ++i) {
    steering_curve.push_back(boost::python::extract<carla::geom::Vector2D &>(list[i]));
  }
  self.steering_curve = steering_curve;
}

boost::python::object VehiclePhysicsControl_init(boost::python::tuple args, boost::python::dict kwargs) {
    // Args names
    const uint32_t NUM_ARGUMENTS = 16;
    const char* args_names[NUM_ARGUMENTS] = {
      "torque_curve",
      "max_rpm",
      "moi",
      "damping_rate_full_throttle",
      "damping_rate_zero_throttle_clutch_engaged",
      "damping_rate_zero_throttle_clutch_disengaged",

      "use_gear_autobox",
      "gear_switch_time",
      "clutch_strength",

      "mass",
      "drag_coefficient",
      "inertia_tensor_scale",

      "center_of_mass",
      "steering_curve",
      "wheels"
    };

    boost::python::object self = args[0];

    auto res = self.attr("__init__")();
    if (len(args) > 1) {
      for (unsigned int i=0; i < len(args)-1; ++i)
        self.attr(args_names[i]) = args[i+1];
    }

    for (unsigned int i = 0; i < NUM_ARGUMENTS; ++i) {
      if (kwargs.contains(args_names[i])) {
        self.attr(args_names[i]) = kwargs[args_names[i]];
      }
    }

    return res;
}

void export_control() {
  using namespace boost::python;
  namespace cr = carla::rpc;
  namespace cg = carla::geom;

  class_<cr::VehicleControl>("VehicleControl")
    .def(init<float, float, float, bool, bool, bool, int>(
        (arg("throttle")=0.0f,
         arg("steer")=0.0f,
         arg("brake")=0.0f,
         arg("hand_brake")=false,
         arg("reverse")=false,
         arg("manual_gear_shift")=false,
         arg("gear")=0)))
    .def_readwrite("throttle", &cr::VehicleControl::throttle)
    .def_readwrite("steer", &cr::VehicleControl::steer)
    .def_readwrite("brake", &cr::VehicleControl::brake)
    .def_readwrite("hand_brake", &cr::VehicleControl::hand_brake)
    .def_readwrite("reverse", &cr::VehicleControl::reverse)
    .def_readwrite("manual_gear_shift", &cr::VehicleControl::manual_gear_shift)
    .def_readwrite("gear", &cr::VehicleControl::gear)
    .def("__eq__", &cr::VehicleControl::operator==)
    .def("__ne__", &cr::VehicleControl::operator!=)
    .def(self_ns::str(self_ns::self))
  ;

  class_<cr::WalkerControl>("WalkerControl")
    .def(init<cg::Vector3D, float, bool>(
        (arg("direction")=cg::Vector3D{1.0f, 0.0f, 0.0f},
         arg("speed")=0.0f,
         arg("jump")=false)))
    .def_readwrite("direction", &cr::WalkerControl::direction)
    .def_readwrite("speed", &cr::WalkerControl::speed)
    .def_readwrite("jump", &cr::WalkerControl::jump)
    .def("__eq__", &cr::WalkerControl::operator==)
    .def("__ne__", &cr::WalkerControl::operator!=)
    .def(self_ns::str(self_ns::self))
  ;

  class_<cr::WheelPhysicsControl>("WheelPhysicsControl")
    .def(init<float, float, float, bool>(
        (arg("tire_friction")=0.0f,
         arg("damping_rate")=0.0f,
         arg("steer_angle")=0.0f,
         arg("disable_steering")=false)))
    .def_readwrite("tire_friction", &cr::WheelPhysicsControl::tire_friction)
    .def_readwrite("damping_rate", &cr::WheelPhysicsControl::damping_rate)
    .def_readwrite("steer_angle", &cr::WheelPhysicsControl::steer_angle)
    .def_readwrite("disable_steering", &cr::WheelPhysicsControl::disable_steering)
    .def("__eq__", &cr::WheelPhysicsControl::operator==)
    .def("__ne__", &cr::WheelPhysicsControl::operator!=)
    .def(self_ns::str(self_ns::self))
  ;

  class_<cr::VehiclePhysicsControl>("VehiclePhysicsControl", no_init)
    .def("__init__", raw_function(VehiclePhysicsControl_init), "raw ctor")
    .def(init<>())

    .add_property("torque_curve", &GetTorqueCurve, &SetTorqueCurve)
    .def_readwrite("max_rpm", &cr::VehiclePhysicsControl::max_rpm)
    .def_readwrite("moi", &cr::VehiclePhysicsControl::moi)
    .def_readwrite("damping_rate_full_throttle", &cr::VehiclePhysicsControl::damping_rate_full_throttle)
    .def_readwrite("damping_rate_zero_throttle_clutch_engaged", &cr::VehiclePhysicsControl::damping_rate_zero_throttle_clutch_engaged)
    .def_readwrite("damping_rate_zero_throttle_clutch_disengaged", &cr::VehiclePhysicsControl::damping_rate_zero_throttle_clutch_disengaged)
    
    .def_readwrite("use_gear_autobox", &cr::VehiclePhysicsControl::use_gear_autobox)
    .def_readwrite("gear_switch_time", &cr::VehiclePhysicsControl::gear_switch_time)
    .def_readwrite("clutch_strength", &cr::VehiclePhysicsControl::clutch_strength)
    
    .def_readwrite("mass", &cr::VehiclePhysicsControl::mass)
    .def_readwrite("drag_coefficient", &cr::VehiclePhysicsControl::drag_coefficient)
    .def_readwrite("inertia_tensor_scale", &cr::VehiclePhysicsControl::inertia_tensor_scale)
    
    .def_readwrite("center_of_mass", &cr::VehiclePhysicsControl::center_of_mass)
    
    .add_property("steering_curve", &GetSteeringCurve, &SetSteeringCurve)
    .add_property("wheels", &GetWheels, &SetWheels)
    
    .def("__eq__", &cr::VehiclePhysicsControl::operator==)
    .def("__ne__", &cr::VehiclePhysicsControl::operator!=)
    .def(self_ns::str(self_ns::self))
  ;
}
