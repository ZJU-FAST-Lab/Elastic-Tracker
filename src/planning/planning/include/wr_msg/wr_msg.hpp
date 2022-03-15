#pragma once
#include <ros/ros.h>
#include <ros/serialization.h>

#include <fstream>

// https://answers.ros.org/question/11289/read-or-write-message-fromto-file/

namespace wr_msg {

template <class MSG, class FILENAME>
static void writeMsg(const MSG& msg, const FILENAME& name) {
  std::ofstream ofs(name, std::ios::out | std::ios::binary);

  uint32_t serial_size = ros::serialization::serializationLength(msg);
  boost::shared_array<uint8_t> obuffer(new uint8_t[serial_size]);

  ros::serialization::OStream ostream(obuffer.get(), serial_size);
  ros::serialization::serialize(ostream, msg);
  ofs.write((char*)obuffer.get(), serial_size);
  ofs.close();
}

template <class MSG, class FILENAME>
static void readMsg(MSG& msg, const FILENAME& name) {
  std::ifstream ifs(name, std::ios::in | std::ios::binary);
  ifs.seekg(0, std::ios::end);
  std::streampos end = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::streampos begin = ifs.tellg();

  uint32_t file_size = end - begin;
  boost::shared_array<uint8_t> ibuffer(new uint8_t[file_size]);
  ifs.read((char*)ibuffer.get(), file_size);
  ros::serialization::IStream istream(ibuffer.get(), file_size);
  ros::serialization::deserialize(istream, msg);
  ifs.close();
}

}  // namespace wr_msg