// Copyright (c) 2021, Lawrence Livermore National Security, LLC.
// See top-level LICENSE file for details.

#include "caliper/CaliperService.h"

#include "../Services.h"

#include "caliper/Caliper.h"
#include "caliper/RegionProfile.h"
#include "caliper/SnapshotRecord.h"

#include "caliper/common/Log.h"
#include "caliper/common/OutputStream.h"
#include "caliper/common/RuntimeConfig.h"

#include "../../common/util/format_util.h"

#include <chrono>
#include <iomanip>
#include <map>
#include <string>

using namespace cali;

namespace
{

std::ostream& write_ldms_record(std::ostream& os, int mpi_rank, RegionProfile& profile)
{
    std::map<std::string, double> region_times;
    double total_time = 0;

    std::tie(region_times, std::ignore, total_time) =
        profile.inclusive_path_profile();

    double unix_ts = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    for (const auto &p : region_times) {
        // ignore regions with < 5% of the epoch's total time
        if (p.second < 0.05 * total_time)
            continue;

        os << "{ \"timestamp\": " << std::fixed << unix_ts
           << ", \"duration\": "  << std::fixed << p.second;

        util::write_esc_string(os << ", \"path\": \"", p.first) << "\"";

        if (mpi_rank >= 0)
            os << ", \"rank\": " << mpi_rank;

        os << " }\n";
    }

    return os;
}

class LdmsForwarder
{
    RegionProfile profile;
    OutputStream  stream;

    std::string   filename;

    void snapshot(Caliper* c, Channel*) {
        Entry e = c->get(c->get_attribute("mpi.rank"));
        int rank = e.empty() ? -1 : e.value().to_int();

        write_ldms_record(*stream.stream(), rank, profile) << "\n";

        profile.clear(); // reset profile - skip to create a cumulative profile
    }

    void post_init(Caliper* c, Channel* channel) {
        std::vector<Entry> rec;
        stream.set_filename(filename.c_str(), *c, rec);
        profile.start();
    }

    LdmsForwarder(const char* fname)
        : filename { fname }
    { }

public:

    static const char* s_spec;

    static void create(Caliper* c, Channel* channel) {
        ConfigSet cfg = 
            services::init_config_from_spec(channel->config(), s_spec);

        LdmsForwarder* instance = new LdmsForwarder(cfg.get("filename").to_string().c_str());

        channel->events().post_init_evt.connect(
            [instance](Caliper* c, Channel* channel){
                instance->post_init(c, channel);
            });
        channel->events().snapshot.connect(
            [instance](Caliper* c, Channel* channel, int scope, SnapshotView, SnapshotBuilder&){
                instance->snapshot(c, channel);
            });
        channel->events().finish_evt.connect(
            [instance](Caliper* c, Channel* chn){
                delete instance;
            });

        Log(1).stream() << channel->name() << "Initialized LDMS forwarder\n";
    }
};

const char* LdmsForwarder::s_spec = R"json(
{   
 "name"        : "ldms",
 "description" : "Forward Caliper regions to LDMS (prototype)",
 "config"      :
 [
  {
    "name"        : "filename",
    "description" : "Output file name, or stdout/stderr",
    "type"        : "string",
    "value"       : "stdout"
  }
 ]
}
)json";

} // namespace [anonymous]

namespace cali
{

CaliperService ldms_service { ::LdmsForwarder::s_spec, ::LdmsForwarder::create };

}
