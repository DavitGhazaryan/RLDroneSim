#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include <gz/transport/Node.hh>
#include <gz/msgs/world_stats.pb.h>
#include <gz/msgs/empty.pb.h>
#include <gz/msgs/double.pb.h>


using namespace std::chrono_literals;

std::atomic<double> g_sim_time_sec{0.0};

void StatsCb(const gz::msgs::WorldStatistics &msg) {
  const auto &t = msg.sim_time();
  double seconds = static_cast<double>(t.sec()) + t.nsec() * 1e-9;
  g_sim_time_sec.store(seconds, std::memory_order_relaxed);
}

bool SimTimeSrv(const gz::msgs::Empty &, gz::msgs::Double &rep) {
  rep.set_data(g_sim_time_sec.load(std::memory_order_relaxed));
  return true;  // success
}

int main(int argc, char **argv) {
  std::string world = (argc > 1) ? argv[1] : "simple_world";
  std::string stats_topic = "/world/" + world + "/stats";
  std::string service_name = "/sim_time";  // change if you want

  gz::transport::Node node;

  // Subscribe
  if (!node.Subscribe(stats_topic, &StatsCb)) {
    std::cerr << "Failed to subscribe: " << stats_topic << std::endl;
    return 1;
  }

  // Advertise service
  if (!node.Advertise(service_name, &SimTimeSrv)) {
    std::cerr << "Failed to advertise service: " << service_name << std::endl;
    return 1;
  }

  std::cout << "Listening on " << stats_topic << "\n";
  std::cout << "Service ready at " << service_name << " (returns gz.msgs.Double)\n";

  // Keep process alive (gz-transport runs callbacks on background threads)
  while (true) std::this_thread::sleep_for(1s);
}
