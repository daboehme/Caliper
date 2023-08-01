// Copyright (c) 2015-2022, Lawrence Livermore National Security, LLC.
// See top-level LICENSE file for details.

#include "caliper/CaliperService.h"
#include "../Services.h"

#include "caliper/Caliper.h"
#include "caliper/SnapshotRecord.h"

#include "caliper/common/Log.h"
#include "caliper/common/RuntimeConfig.h"

#include <pthread.h>
#include <time.h>

#include <array>
#include <chrono>
#include <cmath>

using namespace cali;

namespace
{

inline double get_timestamp()
{
    return 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

class ThreadMonitor
{
    pthread_t monitor_thread;
    bool      thread_running;

    Channel*  channel;

    time_t    m_sleep_sec;
    long      m_sleep_nsec;

    Attribute m_snapshot_attr;
    Attribute m_timestamp_attr;

    unsigned  m_num_snapshots;

    void snapshot() {
        Caliper c;

        std::array<Entry, 2> info = {
            Entry(m_snapshot_attr, cali_make_variant_from_uint(m_num_snapshots++)),
            Entry(m_timestamp_attr, cali_make_variant_from_double(get_timestamp()))
        };

        c.push_snapshot(channel, SnapshotView(info.size(), info.data()));
    }

    // the main monitor loop
    void monitor() {
        struct timespec req { m_sleep_sec, m_sleep_nsec };

        while (true) {
            struct timespec rem = req;
            int ret = 0;

            do {
                ret = nanosleep(&rem, &rem);
            } while (ret == EINTR);

            if (ret != 0) {
                Log(0).perror(errno, "thread_monitor: nanosleep(): ");
                break;
            }

            // disable cancelling during snapshot
            pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, nullptr);
            snapshot();
            pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, nullptr);
        }
    }

    static void* thread_fn(void* p) {
        ThreadMonitor* instance = static_cast<ThreadMonitor*>(p);
        instance->monitor();
        return nullptr;
    }

    bool start() {
        if (pthread_create(&monitor_thread, nullptr, thread_fn, this) != 0) {
            Log(0).stream() << channel->name()
                << ": thread_monitor(): pthread_create() failed\n";
            return false;
        }

        Log(1).stream() << channel->name()
            << ": thread_monitor: monitoring thread initialized\n";

        thread_running = true;
        return true;
    }

    void cancel() {
        Log(2).stream() << channel->name() << ": thread_monitor: cancelling monitoring thread\n";

        if (pthread_cancel(monitor_thread) != 0)
            Log(0).stream() << channel->name() << ": thread_monitor: pthread_cancel() failed\n";

        pthread_join(monitor_thread, nullptr);

        Log(1).stream() << channel->name() << ": thread_monitor: monitoring thread finished\n";
    }

    void post_init_cb() {
        if (!start())
            return;
    }

    void finish_cb() {
        if (thread_running)
            cancel();

        Log(1).stream() << channel->name()
            << ": thread_monitor: triggered " << m_num_snapshots << " snapshots\n";
    }

    ThreadMonitor(Caliper* c, Channel* chn)
        : thread_running(false),
          channel(chn),
          m_sleep_sec(1),
          m_sleep_nsec(0),
          m_num_snapshots(0)
        {
            m_snapshot_attr =
                c->create_attribute("monitor.snapshot", CALI_TYPE_UINT,
                                    CALI_ATTR_ASVALUE |
                                    CALI_ATTR_SKIP_EVENTS);
            m_timestamp_attr =
                c->create_attribute("monitor.timestamp", CALI_TYPE_DOUBLE,
                                    CALI_ATTR_ASVALUE |
                                    CALI_ATTR_SKIP_EVENTS);

            ConfigSet config = services::init_config_from_spec(channel->config(), s_spec);

            double sleep_val = fabs(config.get("time_interval").to_double());

            m_sleep_sec = static_cast<time_t>(floor(sleep_val));
            m_sleep_nsec = static_cast<long>((sleep_val - floor(sleep_val)) * 1e9);

            Log(2).stream() << channel->name() << ": thread_monitor: sleep interval: "
                << m_sleep_sec << " sec, " << m_sleep_nsec << " nsec.\n";
        }

public:

    static const char* s_spec;

    static void create(Caliper* c, Channel* channel) {
        ThreadMonitor* instance = new ThreadMonitor(c, channel);

        channel->events().post_init_evt.connect(
            [instance](Caliper* c, Channel* channel){
                instance->post_init_cb();
            });
        channel->events().finish_evt.connect(
            [instance](Caliper*, Channel*){
                instance->finish_cb();
                delete instance;
            });

        Log(1).stream() << channel->name() << ": Registered thread_monitor service\n";
    }

};

const char* ThreadMonitor::s_spec = R"json(
{   "name": "thread_monitor",
    "description": "Run a measurement thread triggering snapshots at regular intervals",
    "config": [
        {   "name"        : "time_interval",
            "description" : "Length in seconds of the measurement interval",
            "type"        : "double",
            "value"       : "0.1"
        }
    ]
}
)json";

} // namespace [anonymous]

namespace cali
{

CaliperService thread_monitor_service { ::ThreadMonitor::s_spec, ::ThreadMonitor::create };

}
