// Copyright (c) 2015-2022, Lawrence Livermore National Security, LLC.
// See top-level LICENSE file for details.

#include "caliper/CaliperService.h"

#include "caliper/Caliper.h"
#include "caliper/SnapshotRecord.h"

#include "caliper/common/Attribute.h"
#include "caliper/common/Log.h"

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

using namespace cali;

namespace
{

std::pair<uint64_t, uint64_t> parse_statm(const char* buf, ssize_t max)
{
    uint64_t numbers[7] = { 0, 0, 0, 0, 0, 0, 0 };
    int nn = 0;

    for (ssize_t n = 0; n < max && nn < 6; ++n) {
        char c = buf[n];
        if (c >= '0' && c <= '9')
            numbers[nn] = numbers[nn] * 10 + (static_cast<unsigned>(c) - static_cast<unsigned>('0'));
        else if (c == ' ')
            ++nn;
    }

    return std::make_pair(numbers[0], numbers[5]);
}

class MemstatService
{
    Attribute m_vmsize_attr;
    Attribute m_vmdata_attr;

    int       m_fd;

    unsigned  m_failed;

    void snapshot_cb(Caliper*, int scopes, SnapshotBuilder& rec) {
        if (!(scopes & CALI_SCOPE_PROCESS))
            return;

        char buf[80];
        ssize_t ret = pread(m_fd, buf, sizeof(buf), 0);

        if (ret < 0) {
            ++m_failed;
            return;
        }

        auto p = parse_statm(buf, ret);

        rec.append(m_vmsize_attr, cali_make_variant_from_uint(p.first));
        rec.append(m_vmdata_attr, cali_make_variant_from_uint(p.second));
    }

    void finish_cb(Caliper*, Channel* channel) {
        if (m_failed > 0)
            Log(0).stream() << channel->name()
                            << ": memstat: failed to read /proc/self/statm "
                            << m_failed << " times\n";
    }

    MemstatService(Caliper* c, int fd)
        : m_fd     { fd },
          m_failed { 0  }
    {
        m_vmsize_attr =
            c->create_attribute("memstat.vmsize", CALI_TYPE_UINT,
                CALI_ATTR_SCOPE_PROCESS |
                CALI_ATTR_ASVALUE       |
                CALI_ATTR_AGGREGATABLE);
        m_vmdata_attr =
            c->create_attribute("memstat.data", CALI_TYPE_UINT,
                CALI_ATTR_SCOPE_PROCESS |
                CALI_ATTR_ASVALUE       |
                CALI_ATTR_AGGREGATABLE);
    }

public:

    static void memstat_register(Caliper* c, Channel* channel) {
        int fd = open("/proc/self/statm", O_RDONLY);

        if (fd < 0) {
            Log(0).perror(errno, "open(\"/proc/self/statm\")");
            return;
        }

        MemstatService* instance = new MemstatService(c, fd);

        channel->events().snapshot.connect(
            [instance](Caliper* c, Channel*, int scopes, SnapshotView, SnapshotBuilder& rec){
                instance->snapshot_cb(c, scopes, rec);
            });
        channel->events().finish_evt.connect(
            [instance](Caliper* c, Channel* channel){
                instance->finish_cb(c, channel);
                close(instance->m_fd);
                delete instance;
            });

        Log(1).stream() << channel->name() << ": registered memstat service\n";
    }
};

const char* memstat_spec = R"json(
{
    "name"        : "memstat",
    "description" : "Record process memory info from /proc/self/statm"
}
)json";

}

namespace cali
{

CaliperService memstat_service { ::memstat_spec, ::MemstatService::memstat_register };

}