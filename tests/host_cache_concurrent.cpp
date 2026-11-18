// Eight writers are held *after* each cache miss and before publication.
// Each writer uses a different physics fingerprint and a known constant field.
#include "host_grid.h"
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>
extern "C" void* poremc_environment_create(std::size_t,const double*,const double*,const double*,const double*,const double*,double,int,double,int,int,int);
extern "C" void poremc_environment_destroy(void*);

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  constexpr int writers = 8;
  const double zero[3]{}, unit = 1.;
  poremc::HostGridConfig cfg;
  cfg.cell = {24,0,0, 0,24,0, 0,0,24};
  cfg.mesh = {32,32,32};
  cfg.cache_path = argv[1];
  std::mutex mutex;
  std::condition_variable ready;
  int waiting = 0;
  auto make = [&](int id, bool writing) {
    auto local = cfg;
    local.cache_key = "writer-" + std::to_string(id);
    std::unique_ptr<void,decltype(&poremc_environment_destroy)> host(
      poremc_environment_create(0,zero,zero,zero,zero,cfg.cell.data(),9,2,.28,1,2,1), poremc_environment_destroy);
    if (!host) throw std::runtime_error("environment initialization failed");
    auto build = [&] {
      if (!writing) throw std::runtime_error("published file did not pass integrity verification");
      std::unique_lock<std::mutex> guard(mutex);
      ++waiting;
      ready.notify_all();
      if (!ready.wait_for(guard,std::chrono::seconds(30),[&]{return waiting == writers;}))
        throw std::runtime_error("concurrent writer rendezvous timed out (path must be new)");
      return std::vector<double>(32*32*32,double(id + 1));
    };
    poremc::HostGrid grid(local,host.get(),1,zero,zero,&unit,build);
    const auto stats = grid.stats();
    if (grid.site_energy(0,zero) != id + 1 ||
        (writing ? stats.cache_writes : stats.cache_loads) != 1)
      throw std::runtime_error("cache energy/status unexpected");
  };
  try {
    std::vector<std::future<void>> jobs;
    for (int i=0; i<writers; ++i) jobs.push_back(std::async(std::launch::async, make, i, true));
    for (auto& job : jobs) job.get();
    std::ifstream in(cfg.cache_path, std::ios::binary);
    char magic[8]{};
    std::uint32_t bytes = 0;
    in.read(magic,8).read(reinterpret_cast<char*>(&bytes),sizeof(bytes));
    if (!in || std::string(magic,8) != "PMCHG061" || bytes != 8) return 3;
    std::string key(bytes,'\0'); in.read(key.data(),bytes);
    int index = std::stoi(key.substr(7));
    if (index < 0 || index >= writers) return 4;
    make(index, false);
    std::cout << "{\"simultaneous_cold_writers\":" << writers
              << ",\"accepted_cache_hits\":1,\"verified_field_owner\":" << index << "}\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
