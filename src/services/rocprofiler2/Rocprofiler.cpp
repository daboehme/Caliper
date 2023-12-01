// Copyright (c) 2015-2022, Lawrence Livermore National Security, LLC.
// See top-level LICENSE file for details.

#include "caliper/CaliperService.h"

#include "../Services.h"

#include "caliper/Caliper.h"
#include "caliper/SnapshotRecord.h"

#include "caliper/common/Attribute.h"
#include "caliper/common/Log.h"
#include "caliper/common/RuntimeConfig.h"

#include "caliper/common/c-util/unitfmt.h"

#include "../../common/util/demangle.h"

#include <rocprofiler.h>

#include <mutex>
#include <map>

#include <cstring>

using namespace cali;

#define CALL_ROCM(call, msg)                \
    do {                                         \
        auto ret = call;                         \
        if (ret != ROCPROFILER_STATUS_SUCCESS) {   \
            cali::Log(0).stream() << msg << ": " \
                << rocprofiler_error_str(ret);  \
            return;                              \
        }                                        \
    } while (false)

namespace
{

void print_rocm_error(Channel* channel, const char* msg, rocprofiler_status_t s)
{
    Log(0).stream() << channel->name() << ": " << msg << " error: "
        << rocprofiler_error_str(s)
        << "\n";
}

const char* activity_domain_name(rocprofiler_tracer_activity_domain_t id)
{
    switch (id) {
        case ACTIVITY_DOMAIN_HIP_API:
            return "hip_api";
        case ACTIVITY_DOMAIN_HIP_OPS:
            return "hip_ops";
        case ACTIVITY_DOMAIN_HSA_API:
            return "hsa_api";
        case ACTIVITY_DOMAIN_HSA_OPS:
            return "hsa_ops";
        case ACTIVITY_DOMAIN_ROCTX:
            return "roctx";
        default:
            return nullptr;
    }
}

class RocprofilerService
{
    Attribute m_domain_attr;
    Attribute m_activity_attr;
    Attribute m_api_attr;
    Attribute m_kernel_name_attr;
    Attribute m_begin_ts_attr;
    Attribute m_end_ts_attr;
    Attribute m_activity_duration_attr;
    Attribute m_api_duration_attr;
    Attribute m_host_timestamp_attr;
    Attribute m_host_duration_attr;
    Attribute m_host_starttime_attr;
    Attribute m_correlation_attr;

    rocprofiler_session_id_t m_session_id;
    rocprofiler_buffer_id_t  m_buffer_id;

    std::map< uint64_t, std::vector<Entry> > m_correlation_map;
    std::mutex m_correlation_map_mutex;

    bool      m_trace_api;
    bool      m_trace_activities;
    bool      m_correlate_contex;
    bool      m_record_host_timestamps;

    unsigned  m_num_flushes;
    unsigned  m_num_records;

    Channel*  m_channel;

    static RocprofilerService* s_instance;

    void create_attributes(Caliper* c) {
        m_domain_attr =
            c->create_attribute("rocm.domain", CALI_TYPE_STRING,
                CALI_ATTR_DEFAULT | CALI_ATTR_SKIP_EVENTS);
        m_activity_attr =
            c->create_attribute("rocm.activity", CALI_TYPE_STRING,
                CALI_ATTR_DEFAULT | CALI_ATTR_SKIP_EVENTS);
        m_api_attr =
            c->create_attribute("rocm.api", CALI_TYPE_STRING,
                CALI_ATTR_DEFAULT | CALI_ATTR_NESTED);
        m_kernel_name_attr =
            c->create_attribute("rocm.kernel.name", CALI_TYPE_STRING,
                CALI_ATTR_DEFAULT | CALI_ATTR_SKIP_EVENTS);
        m_begin_ts_attr =
            c->create_attribute("rocm.starttime", CALI_TYPE_UINT,
                CALI_ATTR_ASVALUE | CALI_ATTR_SKIP_EVENTS);
        m_end_ts_attr =
            c->create_attribute("rocm.endtime", CALI_TYPE_UINT,
                CALI_ATTR_ASVALUE | CALI_ATTR_SKIP_EVENTS);
        m_activity_duration_attr =
            c->create_attribute("rocm.activity.duration", CALI_TYPE_UINT,
                CALI_ATTR_ASVALUE | CALI_ATTR_SKIP_EVENTS | CALI_ATTR_AGGREGATABLE);
        m_api_duration_attr =
            c->create_attribute("rocm.api.duration", CALI_TYPE_UINT,
                CALI_ATTR_ASVALUE | CALI_ATTR_SKIP_EVENTS | CALI_ATTR_AGGREGATABLE);
        m_correlation_attr =
            c->create_attribute("rocm.correlation_id", CALI_TYPE_UINT,
                CALI_ATTR_ASVALUE | CALI_ATTR_SKIP_EVENTS);

        m_host_timestamp_attr =
            c->create_attribute("rocm.host.timestamp", CALI_TYPE_UINT,
                                CALI_ATTR_SCOPE_THREAD |
                                CALI_ATTR_ASVALUE      |
                                CALI_ATTR_SKIP_EVENTS);
        m_host_duration_attr =
            c->create_attribute("rocm.host.duration", CALI_TYPE_UINT,
                                CALI_ATTR_SCOPE_THREAD |
                                CALI_ATTR_ASVALUE      |
                                CALI_ATTR_SKIP_EVENTS  |
                                CALI_ATTR_AGGREGATABLE);
        m_host_starttime_attr =
            c->create_attribute("rocm.host.starttime", CALI_TYPE_UINT,
                                CALI_ATTR_SCOPE_PROCESS |
                                CALI_ATTR_SKIP_EVENTS);
    }

    void push_correlation(uint64_t id, std::vector<Entry>&& ctx) {
        std::lock_guard<std::mutex>
            g(m_correlation_map_mutex);

        m_correlation_map[id] = ctx;
    }

    std::vector<Entry> pop_correlation(uint64_t id) {
        std::vector<Entry> ret;

        std::lock_guard<std::mutex>
            g(m_correlation_map_mutex);

        auto it = m_correlation_map.find(id);

        if (it != m_correlation_map.end()) {
            ret = it->second;
            m_correlation_map.erase(it);
        }

        return ret;
    }

    void handle_sync_api_trace_record(rocprofiler_record_tracer_t trec, rocprofiler_session_id_t session) {
        if (trec.domain == ACTIVITY_DOMAIN_HIP_API) {
            const char* cptr = nullptr;
            rocprofiler_query_tracer_operation_name(trec.domain, trec.operation_id, &cptr);

            // skip __hipPush/PopCallConfiguration calls
            if (!cptr || strncmp(cptr, "__hip", 5) == 0)
                return;

            Caliper c;

            if (trec.phase == ROCPROFILER_PHASE_ENTER)
                c.begin(m_api_attr, Variant(cptr));
            else if (trec.phase == ROCPROFILER_PHASE_EXIT)
                c.end_with_value_check(m_api_attr, Variant(cptr));
/*
            const Attribute attr[5] = {
                m_domain_attr,
                m_api_attr,
                m_begin_ts_attr,
                m_end_ts_attr,
                m_api_duration_attr
            };
            const Variant data[5] = {
                Variant(CALI_TYPE_STRING, "hip_api", 7),
                Variant(cptr),
                cali_make_variant_from_uint(trec.timestamps.begin.value),
                cali_make_variant_from_uint(trec.timestamps.end.value),
                cali_make_variant_from_uint(duration)
            };

            FixedSizeSnapshotRecord<64> rec;
            Caliper c;

            c.pull_context(rec.builder());

            if (m_correlate_contex) {
                auto view = rec.view();
                push_correlation(trec.correlation_id.value,
                    std::vector<Entry> { view.begin(), view.end() });
            }

            c.make_record(5, attr, data, rec.builder());
            m_channel->events().process_snapshot(&c, m_channel, SnapshotView(), rec.view());
*/
        }
    }

    void handle_trace_record(Caliper& c, rocprofiler_record_tracer_t trec, rocprofiler_session_id_t session) {
        std::string activity_name;
        std::string kernel_name;

        if (trec.domain == ACTIVITY_DOMAIN_HIP_OPS) {
            if (trec.name)
                kernel_name = util::demangle(trec.name);
        }

        cali::Node* node = nullptr;
        const char* domain_name = activity_domain_name(trec.domain);

        if (domain_name)
            node = c.make_tree_entry(m_domain_attr, Variant(domain_name));
        if (activity_name.size() > 0)
            node = c.make_tree_entry(m_activity_attr, Variant(activity_name.c_str()), node);
        if (kernel_name.size() > 0)
            node = c.make_tree_entry(m_kernel_name_attr, Variant(kernel_name.c_str()), node);

        if (node) {
            FixedSizeSnapshotRecord<64> rec;

            rec.builder().append(Entry(node));

            uint64_t duration = trec.timestamps.end.value - trec.timestamps.begin.value;

            rec.builder().append(m_begin_ts_attr, cali_make_variant_from_uint(trec.timestamps.begin.value));
            rec.builder().append(m_end_ts_attr, cali_make_variant_from_uint(trec.timestamps.end.value));
            rec.builder().append(m_activity_duration_attr, cali_make_variant_from_uint(duration));

            if (m_correlate_contex) {
                auto ctx = pop_correlation(trec.correlation_id.value);
                rec.builder().append(ctx.size(), ctx.data());
            }

            m_channel->events().process_snapshot(&c, m_channel, SnapshotView(), rec.view());
        }
    }

    void flush_records(const rocprofiler_record_header_t* record, const rocprofiler_record_header_t* end,
            rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
        unsigned count = 0;

        Caliper c;

        for ( ; record < end; rocprofiler_next_record(record, &record, session_id, buffer_id)) {
            switch (record->kind) {
            case ROCPROFILER_TRACER_RECORD:
            {
                auto* trec = reinterpret_cast<const rocprofiler_record_tracer_t*>(record);
                handle_trace_record(c, *trec, session_id);
                ++count;
                break;
            }
            default:
                break;
            }
        }

        m_num_records += count;
        ++m_num_flushes;

        Log(1).stream() << m_channel->name() << ": rocprofiler: flushed " << count << " records\n";
    }

    void init_api_tracing(Channel* channel) {
        CALL_ROCM(rocprofiler_create_buffer(m_session_id,
            [](const rocprofiler_record_header_t* record, const rocprofiler_record_header_t* end_record,
                rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
                    s_instance->flush_records(record, end_record, session_id, buffer_id);
                },
            0x800000, &m_buffer_id),
            "rocprofiler_create_buffer()");

        rocprofiler_tracer_activity_domain_t filters[2] = {
            ACTIVITY_DOMAIN_HIP_API,
            ACTIVITY_DOMAIN_HIP_OPS
        };

        size_t f_count = 2;
        rocprofiler_filter_data_t f_data { filters };

        if (m_trace_api && !m_trace_activities) {
            f_count = 1;
        } else if (!m_trace_api && m_trace_activities) {
            f_data = rocprofiler_filter_data_t { &filters[1] };
            f_count = 1;
        } else if (!m_trace_api && !m_trace_activities) {
            f_count = 0;
        }

        rocprofiler_filter_id_t api_filter_id;
        CALL_ROCM(rocprofiler_create_filter(m_session_id, ROCPROFILER_API_TRACE,
            f_data, f_count, &api_filter_id, rocprofiler_filter_property_t {}),
            "rocprofiler_create_filter(ROCPROFILER_API_TRACE)");
        CALL_ROCM(rocprofiler_set_filter_buffer(m_session_id, api_filter_id, m_buffer_id),
            "rocprofiler_set_filter_buffer()");

        CALL_ROCM(rocprofiler_set_api_trace_sync_callback(m_session_id, api_filter_id,
            [](rocprofiler_record_tracer_t rec, rocprofiler_session_id_t session){
                s_instance->handle_sync_api_trace_record(rec, session);
            }),
            "rocprofiler_set_api_trace_sync_callback()");

        if (m_trace_activities) {
            rocprofiler_filter_id_t kernel_filter_id;
            CALL_ROCM(rocprofiler_create_filter(m_session_id, ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION,
                rocprofiler_filter_data_t {}, 0, &kernel_filter_id, rocprofiler_filter_property_t {}),
                "rocprofiler_create_filter(ROCPROFILER_DISPATCH_TIMESTAMP_COLLECTION)");
            CALL_ROCM(rocprofiler_set_filter_buffer(m_session_id, kernel_filter_id, m_buffer_id),
                "rocprofiler_set_filter_buffer(kernel_filter)");
        }

        Log(1).stream() << channel->name() << ": rocprofiler: API tracing session initialized\n";
    }

    void snapshot_cb(Caliper* c, SnapshotBuilder& rec) {
        rocprofiler_timestamp_t timestamp;
        rocprofiler_get_timestamp(&timestamp);

        Variant v_now(cali_make_variant_from_uint(timestamp.value));
        Variant v_prev = c->exchange(m_host_timestamp_attr, v_now);

        rec.append(Entry(m_host_duration_attr,
                   Variant(cali_make_variant_from_uint(timestamp.value - v_prev.to_uint()))));
    }

    void post_init_cb(Caliper* c, Channel* channel) {
        auto ret = rocprofiler_create_session(ROCPROFILER_NONE_REPLAY_MODE, &m_session_id);
        if (ret != ROCPROFILER_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocprofiler_create_session()", ret);
            return;
        }

        init_api_tracing(channel);
        ret = rocprofiler_start_session(m_session_id);
        if (ret != ROCPROFILER_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocprofiler_start_session()", ret);
            return;
        }

        if (m_record_host_timestamps) {
            rocprofiler_timestamp_t timestamp { 0 };
            rocprofiler_get_timestamp(&timestamp);

            c->set(m_host_starttime_attr, cali_make_variant_from_uint(timestamp.value));
            c->set(m_host_timestamp_attr, cali_make_variant_from_uint(timestamp.value));

            channel->events().snapshot.connect(
                [](Caliper* c, Channel*, SnapshotView, SnapshotBuilder& rec){
                    s_instance->snapshot_cb(c, rec);
                });
        }

        Log(1).stream() << channel->name() << ": rocprofiler: session started\n";
    }

    void pre_flush_cb(Caliper*, Channel* channel) {
        auto ret = rocprofiler_flush_data(m_session_id, m_buffer_id);
        if (ret != ROCPROFILER_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocprofiler_flush_buffer()", ret);
        }
    }

    void pre_finish_cb(Caliper* c, Channel* channel) {
        auto ret = rocprofiler_terminate_session(m_session_id);
        if (ret != ROCPROFILER_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocprofiler_terminate_session()", ret);
        }

        Log(1).stream() << channel->name() << ": rocprofiler: session terminated\n";
    }

    void finish_cb(Caliper* c, Channel* channel) {
        rocprofiler_finalize();

        Log(1).stream() << channel->name() << ": rocprofiler: flushed "
            << m_num_records << " records in " << m_num_flushes << " flushes\n";
    }

    RocprofilerService(Caliper* c, Channel* channel)
        : m_num_flushes { 0 },
          m_num_records { 0 },
          m_channel     { channel }
    {
        create_attributes(c);

        auto config = services::init_config_from_spec(channel->config(), s_spec);

        m_trace_api = config.get("trace_api").to_bool();
        m_trace_activities =
            config.get("trace_activities").to_bool();
        m_correlate_contex =
            config.get("correlate_context").to_bool() && m_trace_activities && m_trace_api;
        m_record_host_timestamps =
            config.get("record_host_timestamps").to_bool();
    }

public:

    static const char* s_spec;

    static void register_rocprofiler(Caliper* c, Channel* channel) {
        CALL_ROCM(rocprofiler_initialize(), "rocprofiler_initialize()");

        s_instance = new RocprofilerService(c, channel);

        channel->events().post_init_evt.connect(
            [](Caliper* c, Channel* channel){
                s_instance->post_init_cb(c, channel);
            });
        channel->events().pre_flush_evt.connect(
            [](Caliper* c, Channel* channel, SnapshotView){
                s_instance->pre_flush_cb(c, channel);
            });
        channel->events().pre_finish_evt.connect(
            [](Caliper* c, Channel* channel){
                s_instance->pre_finish_cb(c, channel);
            });
        channel->events().finish_evt.connect(
            [](Caliper* c, Channel* channel){
                s_instance->finish_cb(c, channel);
                delete s_instance;
                s_instance = nullptr;
            });

        Log(1).stream() << channel->name() << ": Registered rocprofiler service."
            << " Using rocprofiler " << rocprofiler_version_major()
            << "." << rocprofiler_version_minor() << std::endl;
    }
};

RocprofilerService* RocprofilerService::s_instance = nullptr;

const char* RocprofilerService::s_spec = R"json(
{   "name": "rocprofiler",
    "description": "Record ROCm API and GPU activities",
    "config": [
        {   "name": "trace_activities",
            "type": "bool",
            "description": "Enable ROCm GPU activity tracing",
            "value": "true"
        },
        {   "name": "trace_api",
            "type": "bool",
            "description": "Enable HIP API tracing",
            "value": "true"
        },
        {   "name": "correlate_context",
            "type": "bool",
            "description": "Correlate HIP activities to host-side context",
            "value": "true"
        },
        {   "name": "record_host_timestamps",
            "type": "bool",
            "description": "Record host-side timestamps with ROCm",
            "value": "false"
        }
    ]
}
)json";

} // namespace [anonymous]

namespace cali
{

CaliperService rocprofiler_service { ::RocprofilerService::s_spec, ::RocprofilerService::register_rocprofiler };

}
