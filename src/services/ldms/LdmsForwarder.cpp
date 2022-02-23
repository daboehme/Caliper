// Copyright (c) 2021, Lawrence Livermore National Security, LLC.
// See top-level LICENSE file for details.

#include "caliper/CaliperService.h"

#include "caliper/Caliper.h"
#include "caliper/RegionProfile.h"
#include "caliper/SnapshotRecord.h"

#include "caliper/common/Log.h"
#include "caliper/common/OutputStream.h"
#include "caliper/common/RuntimeConfig.h"

#include "../../common/util/format_util.h"

#include <map>
#include <string>

using namespace cali;

namespace
{

std::ostream& write_to_json(std::ostream& os, int rank, std::map<std::string, double> region_times) {
    os << "{ ";

    int c = 0;

    if (rank >= 0) {
        os << "\"rank\": " << rank;
        ++c;
    }

    for (auto p : region_times)
        util::write_esc_string(os << (c++ > 0 ? ", \"" : "\""), p.first) << "\": " << p.second;

    return os << " }";
}

class LdmsForwarder
{
    static const ConfigSet::Entry s_configdata[];

    RegionProfile profile;
    OutputStream  stream;

    std::string   filename;

    void snapshot(Caliper* c, Channel*) {
        std::map<std::string, double> region_times;

        std::tie(region_times, std::ignore, std::ignore) =
            profile.inclusive_region_times();

        Entry e = c->get(c->get_attribute("mpi.rank"));
        int rank = e.is_empty() ? -1 : e.value().to_int();

        write_to_json(*stream.stream(), rank, region_times) << "\n";

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

    static void create(Caliper* c, Channel* channel) {
        ConfigSet cfg = channel->config().init("ldms", s_configdata);

        LdmsForwarder* instance = new LdmsForwarder(cfg.get("filename").to_string().c_str());

        channel->events().post_init_evt.connect(
            [instance](Caliper* c, Channel* channel){
                instance->post_init(c, channel);
            });
        channel->events().snapshot.connect(
            [instance](Caliper* c, Channel* channel, int scope, const SnapshotRecord* info, SnapshotRecord* snapshot){
                instance->snapshot(c, channel);
            });
        channel->events().finish_evt.connect(
            [instance](Caliper* c, Channel* chn){
                delete instance;
            });

    }
};

const ConfigSet::Entry LdmsForwarder::s_configdata[] = {
    { "filename", CALI_TYPE_STRING, "stdout",
      "Output file name.",
      "Output file name."
    },
    ConfigSet::Terminator
};

} // namespace [anonymous]

namespace cali
{

CaliperService ldms_service { "ldms", ::LdmsForwarder::create };

}
