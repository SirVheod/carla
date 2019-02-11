// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "Replayer.h"
#include "Recorder.h"
#include "carla/Logging.h"

#include <ctime>
#include <sstream>

namespace carla {
namespace recorder {

Replayer::Replayer() {
}

Replayer::~Replayer() {
  stop();
}

// callbacks
void Replayer::setCallbackEventAdd(RecorderCallbackEventAdd f) {
  callbackEventAdd = std::move(f);
}
void Replayer::setCallbackEventDel(RecorderCallbackEventDel f) {
  callbackEventDel = std::move(f);
}
void Replayer::setCallbackEventParent(RecorderCallbackEventParent f) {
  callbackEventParent = std::move(f);
}
void Replayer::setCallbackEventPosition(RecorderCallbackPosition f) {
  callbackPosition = std::move(f);
}
void Replayer::setCallbackEventFinish(RecorderCallbackFinish f) {
  callbackFinish = std::move(f);
}
void Replayer::setCallbackStateTrafficLight(RecorderCallbackStateTrafficLight f) {
  callbackStateTrafficLight = std::move(f);
}

void Replayer::stop(bool keepActors) {
  if (enabled) {
    enabled = false;
    if (!keepActors)
      processToTime(totalTime);
    file.close();
    // callback
    if (callbackFinish)
      callbackFinish(keepActors);
  }
  if (!keepActors)
    log_warning("Replayer stop");
  else
    log_warning("Replayer stop (keeping actors)");
}

bool Replayer::readHeader() {
  if (file.eof()) {
    return false;
  }

  readValue<char>(file, header.id);
  readValue<uint32_t>(file, header.size);

  return true;
}

void Replayer::skipPacket() {
  file.seekg(header.size, std::ios::cur);
}

std::string Replayer::getInfo(std::string filename) {
  std::stringstream info;

  // try to open
  file.open(filename, std::ios::binary);
  if (!file.is_open()) {
    info << "File " << filename << " not found on server\n";
    return info.str();
  }

  uint16_t i, total;
  RecorderEventAdd eventAdd;
  RecorderEventDel eventDel;
  RecorderEventParent eventParent;
  // RecorderStateTrafficLight stateTraffic;
  bool bShowFrame;

  // read info
  recInfo.read(file);

  // check magic string
  std::string magic;
  magic.resize(recInfo.magic.size());
  std::copy(recInfo.magic.begin(), recInfo.magic.end(), magic.begin());
  if (magic != "CARLA_RECORDER") {
    info << "File is not a CARLA recorder" << std::endl;
    return info.str();
  }

  // show general info
  info << "Version: " << recInfo.version << std::endl;
  info << "Map: " << recInfo.mapfile.data() << std::endl;
  tm *timeInfo = localtime(&recInfo.date);
  char dateStr[100];
  strftime(dateStr, 100, "%x %X", timeInfo);
  info << "Date: " << dateStr << std::endl << std::endl;

  // parse only frames
  while (file) {
    // get header
    if (!readHeader()) {
      break;
    }

    // check for a frame packet
    switch (header.id) {
      case static_cast<char>(RecorderPacketId::Frame):
        frame.read(file);
        // info << "Frame " << frame.id << " at " << frame.elapsed << "
        // seconds\n";
        break;

      case static_cast<char>(RecorderPacketId::Event):
        bShowFrame = true;
        readValue<uint16_t>(file, total);
        if (total > 0 && bShowFrame) {
          info << "Frame " << frame.id << " at " << frame.elapsed << " seconds\n";
          bShowFrame = false;
        }
        for (i = 0; i < total; ++i) {
          eventAdd.read(file);
          // convert buffer to string to show
          std::string s("");
          s.resize(eventAdd.description.id.size());
          std::copy(eventAdd.description.id.begin(), eventAdd.description.id.end(), s.begin());
          info << " Create " << eventAdd.databaseId << ": " << s.data() << " (" <<
              eventAdd.description.uid << ") at (" << eventAdd.transform.location.x << ", " <<
              eventAdd.transform.location.y << ", " << eventAdd.transform.location.z << ")" << std::endl;
          for (auto &att : eventAdd.description.attributes) {
            std::string s1(""), s2("");
            s1.resize(att.id.size());
            std::copy(att.id.begin(), att.id.end(), s1.begin());
            s2.resize(att.value.size());
            std::copy(att.value.begin(), att.value.end(), s2.begin());
            info << "  " << s1.data() << " = " << s2.data() << std::endl;
          }
        }
        readValue<uint16_t>(file, total);
        if (total > 0 && bShowFrame) {
          info << "Frame " << frame.id << " at " << frame.elapsed << " seconds\n";
          bShowFrame = false;
        }
        for (i = 0; i < total; ++i) {
          eventDel.read(file);
          info << " Destroy " << eventDel.databaseId << "\n";
        }
        readValue<uint16_t>(file, total);
        if (total > 0 && bShowFrame) {
          info << "Frame " << frame.id << " at " << frame.elapsed << " seconds\n";
          bShowFrame = false;
        }
        for (i = 0; i < total; ++i) {
          eventParent.read(file);
          info << " Parenting " << eventParent.databaseId << " with " << eventDel.databaseId <<
              " (parent)\n";
        }
        break;

      case static_cast<char>(RecorderPacketId::Position):
        // info << "Positions\n";
        skipPacket();
        break;

      case static_cast<char>(RecorderPacketId::State):
        skipPacket();
        // bShowFrame = true;
        // readValue<uint16_t>(file, total);
        //if (total > 0 && bShowFrame) {
        //  info << "Frame " << frame.id << " at " << frame.elapsed << " seconds\n";
        //  bShowFrame = false;
        //}
        //info << " State traffic lights: " << total << std::endl;
        //for (i = 0; i < total; ++i) {
        //  stateTraffic.read(file);
        //  info << "  Id: " << stateTraffic.databaseId << " state: " << static_cast<char>(0x30 + stateTraffic.state) << " frozen: " << stateTraffic.isFrozen << " elapsedTime: " << stateTraffic.elapsedTime << std::endl;
        //  }
        break;

      default:
        // skip packet
        info << "Unknown packet id: " << header.id << " at offset " << file.tellg() << std::endl;
        skipPacket();
        break;
    }
  }

  info << "\nFrames: " << frame.id << "\n";
  info << "Duration: " << frame.elapsed << " seconds\n";

  file.close();

  return info.str();
}

void Replayer::rewind(void) {
  currentTime = 0.0f;
  totalTime = 0.0f;
  timeToStop = 0.0f;

  file.clear();
  file.seekg(0, std::ios::beg);

  // mark as header as invalid to force reload a new one next time
  frame.elapsed = -1.0f;
  frame.durationThis = 0.0f;

  mappedId.clear();

  // read geneal info
  recInfo.read(file);

  // log_warning("Replayer rewind");
}

// read last frame in file and return the total time recorded
double  Replayer::getTotalTime(void) {

  std::streampos current = file.tellg();

  // parse only frames
  while (file) {
    // get header
    if (!readHeader())
      break;

    // check for a frame packet
    switch (header.id) {
      case static_cast<char>(RecorderPacketId::Frame):
        frame.read(file);
        break;
      default:
        skipPacket();
        break;
    }
  }

  file.clear();
  file.seekg(current, std::ios::beg);
  return frame.elapsed;
}

std::string Replayer::replayFile(std::string filename, double timeStart, double duration) {
  std::stringstream info;
  std::string s;

  // check to stop if we are replaying another
  if (enabled) {
    stop();
  }

  info << "Replaying file: " << filename << std::endl;

  // try to open
  file.open(filename, std::ios::binary);
  if (!file.is_open()) {
    info << "File " << filename << " not found on server\n";
    return info.str();
  }

  // from start
  rewind();

  // get total time of recorder
  totalTime = getTotalTime();
  info << "Total time recorded: " << totalTime << std::endl;
  // set time to start replayer
  if (timeStart < 0.0f) {
    timeStart = totalTime + timeStart;
    if (timeStart < 0.0f) timeStart = 0.0f;
  }
  // set time to stop replayer
  if (duration > 0.0f)
    timeToStop = timeStart + duration;
  else
    timeToStop = totalTime;
  info << "Replaying from " << timeStart << " s - " << timeToStop << " s (" << totalTime << " s)" << std::endl;

  // process all events until the time
  processToTime(timeStart);

  // mark as enabled
  enabled = true;

  return info.str();
}

void Replayer::processToTime(double time) {
  double per = 0.0f;
  double newTime = currentTime + time;
  bool frameFound = false;

  // check if we are in the right frame
  if (newTime >= frame.elapsed && newTime < frame.elapsed + frame.durationThis) {
    per = (newTime - frame.elapsed) / frame.durationThis;
    frameFound = true;
  }

  // process all frames until time we want or end
  while (!file.eof() && !frameFound) {

    // get header
    readHeader();
    // check it is a frame packet
    if (header.id != static_cast<char>(RecorderPacketId::Frame)) {
      if (!file.eof())
        log_error("Replayer file error: waitting for a Frame packet");
      stop();
      break;
    }
    // read current frame
    frame.read(file);

    // check if target time is in this frame
    if (frame.elapsed + frame.durationThis < newTime) {
      per = 0.0f;
    } else {
      per = (newTime - frame.elapsed) / frame.durationThis;
      frameFound = true;
    }

    // info << "Frame: " << frame.id << " (" << frame.durationThis << " / " <<
    // frame.elapsed << ") per: " << per << std::endl;

    // get header
    readHeader();
    // check it is an events packet
    if (header.id != static_cast<char>(RecorderPacketId::Event)) {
      log_error("Replayer file error: waitting for an Event packet");
      stop();
      break;
    }
    processEvents();

    // get header
    readHeader();
    // check it is a positions packet
    if (header.id != static_cast<char>(RecorderPacketId::Position)) {
      log_error("Replayer file error: waitting for a Position packet");
      stop();
      break;
    }
    if (frameFound) {
      processPositions();
    } else {
      skipPacket();
    }

    // get header
    readHeader();
    // check it is an state packet
    if (header.id != static_cast<char>(RecorderPacketId::State)) {
      log_error("Replayer file error: waitting for an State packet");
      stop();
      break;
    }
    if (frameFound) {
      processStates();
    } else {
      skipPacket();
    }

    // log_warning("Replayer new frame");
  }

  // update all positions
  if (enabled && frameFound)
    updatePositions(per);

  // save current time
  currentTime = newTime;

  // stop replay?
  if (currentTime >= timeToStop) {
    // check if we need to stop the replayer and let it continue in simulation mode
    if (timeToStop == totalTime)
      stop();
    else
      stop(true); // keep actors in scene so they continue with AI
  }
}

void Replayer::processEvents(void) {
  uint16_t i, total;
  RecorderEventAdd eventAdd;
  RecorderEventDel eventDel;
  RecorderEventParent eventParent;
  std::stringstream info;

  // create events
  readValue<uint16_t>(file, total);
  for (i = 0; i < total; ++i) {
    std::string s;
    eventAdd.read(file);

    // avoid sensor events
    if (memcmp(eventAdd.description.id.data(), "sensor.", 7) != 0) {

      // show log
      s.resize(eventAdd.description.id.size());
      std::copy(eventAdd.description.id.begin(), eventAdd.description.id.end(), s.begin());
      info.str("");
      info << "Create " << eventAdd.databaseId << " (" << eventAdd.description.uid << ") " << s.data() <<
            std::endl;
      for (const auto &att : eventAdd.description.attributes) {
        std::string s2;
        s.resize(att.id.size());
        std::copy(att.id.begin(), att.id.end(), s.begin());
        s2.resize(att.value.size());
        std::copy(att.value.begin(), att.value.end(), s2.begin());
        info << "  " << s.data() << " = " << s2.data() << std::endl;
      }
      log_warning(info.str());

      // callback
      if (callbackEventAdd) {
        // log_warning("calling callback add");
        auto result = callbackEventAdd(eventAdd.transform,
            std::move(eventAdd.description),
            eventAdd.databaseId);
        switch (result.first) {
          case 0:
            log_warning("actor could not be created");
            break;
          case 1:
            if (result.second != eventAdd.databaseId) {
              log_warning("actor created but with different id");
            }
            // mapping id (recorded Id is a new Id in replayer)
            mappedId[eventAdd.databaseId] = result.second;
            break;

          case 2:
            log_warning("actor already exist, not created");
            // mapping id (say desired Id is mapped to what)
            mappedId[eventAdd.databaseId] = result.second;
            break;
        }

      } else {
        log_warning("callback add is not defined");
      }
    }
  }


  // destroy events
  readValue<uint16_t>(file, total);
  for (i = 0; i < total; ++i) {
    eventDel.read(file);
    info.str("");
    info << "Destroy " << mappedId[eventDel.databaseId] << "\n";
    log_warning(info.str());
    // callback
    if (callbackEventDel) {
      callbackEventDel(mappedId[eventDel.databaseId]);
      mappedId.erase(eventDel.databaseId);
    } else {
      log_warning("callback del is not defined");
    }
  }

  // parenting events
  readValue<uint16_t>(file, total);
  for (i = 0; i < total; ++i) {
    eventParent.read(file);
    info.str("");
    info << "Parenting " << mappedId[eventParent.databaseId] << " with " << mappedId[eventDel.databaseId] <<
          " (parent)\n";
    log_warning(info.str());
    // callback
    if (callbackEventParent) {
      callbackEventParent(mappedId[eventParent.databaseId], mappedId[eventParent.databaseIdParent]);
    } else {
      log_warning("callback parent is not defined");
    }
  }
}

void Replayer::processStates(void) {
  uint16_t i, total;
  RecorderStateTrafficLight stateTrafficLight;
  std::stringstream info;

  // read total traffic light states
  readValue<uint16_t>(file, total);
  for (i = 0; i < total; ++i) {
    stateTrafficLight.read(file);

    // callback
    if (callbackStateTrafficLight) {
      // log_warning("calling callback add");
      stateTrafficLight.databaseId = mappedId[stateTrafficLight.databaseId];
      if (!callbackStateTrafficLight(stateTrafficLight))
        log_warning("callback state traffic light %d called but didn't work", stateTrafficLight.databaseId);
    } else {
      log_warning("callback state traffic light is not defined");
    }
  }
}

void Replayer::processPositions(void) {
  uint16_t i, total;

  // save current as previous
  prevPos = std::move(currPos);

  // read all positions
  readValue<uint16_t>(file, total);
  currPos.clear();
  currPos.reserve(total);
  for (i = 0; i < total; ++i) {
    RecorderPosition pos;
    pos.read(file);
    // assign mapped Id
    auto newId = mappedId.find(pos.databaseId);
    if (newId != mappedId.end()) {
      pos.databaseId = newId->second;
    }
    currPos.push_back(std::move(pos));
  }
}

void Replayer::updatePositions(double per) {
  unsigned int i;
  std::unordered_map<int, int> tempMap;

  // map the id of all previous positions to its index
  for (i = 0; i < prevPos.size(); ++i) {
    tempMap[prevPos[i].databaseId] = i;
  }

  // go through each actor and update
  for (i = 0; i < currPos.size(); ++i) {
    // check if exist a previous position
    auto result = tempMap.find(currPos[i].databaseId);
    if (result != tempMap.end()) {
      // interpolate
      interpolatePosition(prevPos[result->second], currPos[i], per);
    } else {
      // assign last position (we don't have previous one)
      interpolatePosition(currPos[i], currPos[i], 0);
      // log_warning("Interpolation not possible, only one position");
    }
  }
}

// interpolate a position (transform, velocity...)
void Replayer::interpolatePosition(
    const RecorderPosition &prevPos,
    const RecorderPosition &currPos,
    double per) {
  // call the callback
  callbackPosition(prevPos, currPos, per);
}

// tick for the replayer
void Replayer::tick(float delta) {
  // check if there are events to process
  if (enabled) {
    processToTime(delta);
  }

  // log_warning("Replayer tick");
}

} // namespace recorder
} // namespace carla
