// Copyright (c) 2019, Lawrence Livermore National Security, LLC.
// See top-level LICENSE file for details.

#include "caliper/caliper-config.h"

#include "caliper/ChannelController.h"
#include "caliper/ConfigManager.h"

#include "caliper/common/Log.h"
#include "caliper/common/StringConverter.h"

#include "../../services/Services.h"

#include <algorithm>
#include <set>
#include <tuple>

using namespace cali;

namespace
{

class HatchetSampleProfileController : public cali::ChannelController
{
public:

    HatchetSampleProfileController(const cali::ConfigManager::Options& opts, const std::string& format)
        : ChannelController("hatchet-sample-profile", 0, {
                { "CALI_CHANNEL_FLUSH_ON_EXIT", "false" },
                { "CALI_SERVICES_ENABLE", "sampler,trace" }
            })
        {
            config()["CALI_SAMPLER_FREQUENCY"] = opts.get("sampler.frequency", "200").to_string();

            std::string select  = "*,count()";
            std::string groupby = "prop:nested";

            std::string output(opts.get("output", "sample_profile").to_string());

            if (output != "stdout" && output != "stderr") {
                auto pos = output.find_last_of('.');
                std::string ext = (format == "cali" ? ".cali" : ".json");

                if (pos == std::string::npos || output.substr(pos) != ext)
                    output.append(ext);
            }

            auto avail_services = Services::get_available_services();
            bool have_mpi =
                std::find(avail_services.begin(), avail_services.end(), "mpireport")    != avail_services.end();
            bool have_adiak =
                std::find(avail_services.begin(), avail_services.end(), "adiak_import") != avail_services.end();

            if (have_adiak)
                config()["CALI_SERVICES_ENABLE"].append(",adiak_import");

            if (have_mpi) {
                groupby += ",mpi.rank";

                config()["CALI_SERVICES_ENABLE"   ].append(",mpi,mpireport");
                config()["CALI_MPIREPORT_FILENAME"] = output;
                config()["CALI_MPIREPORT_WRITE_ON_FINALIZE"] = "false";
                config()["CALI_MPIREPORT_CONFIG"  ] =
                    std::string("select ") 
                    + opts.query_select("local", "*,count()") 
                    + " group by " 
                    + opts.query_groupby("local", "prop:nested,mpi.rank")
                    + " format " + format;
            } else {
                config()["CALI_SERVICES_ENABLE"   ].append(",report");
                config()["CALI_REPORT_FILENAME"   ] = output;
                config()["CALI_REPORT_CONFIG"     ] =
                    std::string("select ") 
                    + opts.query_select("local", "*,count()")
                    + " group by " 
                    + opts.query_groupby("local", "prop:nested")
                    + " format " + format;
            }

            opts.update_channel_config(config());
        }
};

std::string
check_args(const cali::ConfigManager::Options& opts) {
    Services::add_default_services();
    auto svcs = Services::get_available_services();

    //
    // Check if the sampler service is there
    //

    if (std::find(svcs.begin(), svcs.end(), "sampler") == svcs.end())
        return "hatchet-sample-profile: sampler service is not available";

    //
    // Check if output.format is valid
    //

    std::string format = opts.get("output.format", "json-split").to_string();
    std::set<std::string> allowed_formats = { "cali", "json", "json-split" };

    if (allowed_formats.find(format) == allowed_formats.end())
        return std::string("hatchet-sample-profile: Invalid output format \"") + format + "\"";

    return "";
}

cali::ChannelController*
make_controller(const cali::ConfigManager::Options& opts)
{
    std::string format = opts.get("output.format", "json-split").to_string();

    if (format == "hatchet")
        format = "json-split";

    if (!(format == "json-split" || format == "json" || format == "cali")) {
        format = "json-split";
        Log(0).stream() << "hatchet-region-profile: Unknown output format \"" << format
                        << "\". Using json-split."
                        << std::endl;
    }

    return new HatchetSampleProfileController(opts, format);
}

const char* controller_spec =
    "{"
    " \"name\"        : \"hatchet-sample-profile\","
    " \"description\" : \"Record a sampling profile for processing with hatchet\","
    " \"services\"    : [ \"sampler\" ],"
    " \"categories\"  : [ \"output\"  ],"
    " \"options\": "
    " ["
    "  { "
    "    \"name\": \"output.format\","
    "    \"type\": \"string\","
    "    \"description\": \"Output format ('hatchet', 'cali', 'json')\""
    "  },"
    "  { "
    "    \"name\": \"sample.frequency\","
    "    \"type\": \"int\","
    "    \"description\": \"Sampling frequency in Hz. Default: 200\""
    "  },"
    "  { "
    "    \"name\": \"sample.threads\","
    "    \"type\": \"bool\","
    "    \"description\": \"Profile all threads.\","
    "    \"services\": [ \"pthread\" ]"
    "  },"
    "  { "
    "    \"name\": \"sample.callpath\","
    "    \"type\": \"bool\","
    "    \"description\": \"Perform call-stack unwinding\","
    "    \"services\": [ \"callpath\", \"symbollookup\" ],"
    "    \"extra_config_flags\": { \"CALI_CALLPATH_SKIP_FRAMES\": \"4\" },"
    "    \"query args\": [ { \"level\": \"local\", \"group by\": source.function#callpath.address } ]"
    "  },"
    "  { "
    "    \"name\": \"lookup.module\","
    "    \"type\": \"bool\","
    "    \"description\": \"Lookup source module (.so/.exe)\","
    "    \"services\": [ \"symbollookup\" ],"
    "    \"extra_config_flags\": { \"CALI_SYMBOLLOOKUP_LOOKUP_MODULE\": \"true\" },"
    "    \"query args\": [ { \"level\": \"local\", \"group by\": \"module#cali.sampler.pc\" } ]"
    "  },"
    "  { "
    "    \"name\": \"lookup.sourceloc\","
    "    \"type\": \"bool\","
    "    \"description\": \"Lookup source location (file+line)\","
    "    \"services\": [ \"symbollookup\" ],"
    "    \"extra_config_flags\": { \"CALI_SYMBOLLOOKUP_LOOKUP_SOURCELOC\": \"true\" },"
    "    \"query args\": [ { \"level\": \"local\", \"group by\": \"sourceloc#cali.sampler.pc\" } ]"
    "  }"
    " ]"
    "}";

} // namespace [anonymous]

namespace cali
{

ConfigManager::ConfigInfo hatchet_sample_profile_controller_info
{
    ::controller_spec, ::make_controller, ::check_args
};

}
