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

#include <rocmtools/rocmtools.h>

#include <mutex>
#include <map>

#include <cstring>

using namespace cali;

namespace
{

void print_rocm_error(Channel* channel, const char* msg, rocmtools_status_t s)
{
    Log(0).stream() << channel->name() << ": " << msg << " error: "
        << rocmtools_error_string(s)
        << "\n";
}

const char* activity_domain_name(rocmtools_tracer_activity_domain_t id)
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

class RocmService
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

    rocmtools_session_id_t m_session_id;
    rocmtools_buffer_id_t  m_buffer_id;

    std::map< uint64_t, std::vector<Entry> > m_correlation_map;
    std::mutex m_correlation_map_mutex;

    bool      m_trace_api;
    bool      m_trace_activities;
    bool      m_correlate_contex;
    bool      m_record_host_timestamps;

    unsigned  m_num_flushes;
    unsigned  m_num_records;

    Channel*  m_channel;

    static RocmService* s_instance;

    void create_attributes(Caliper* c) {
        m_domain_attr =
            c->create_attribute("rocm.domain", CALI_TYPE_STRING,
                CALI_ATTR_DEFAULT | CALI_ATTR_SKIP_EVENTS);
        m_activity_attr =
            c->create_attribute("rocm.activity", CALI_TYPE_STRING,
                CALI_ATTR_DEFAULT | CALI_ATTR_SKIP_EVENTS);
        m_api_attr =
            c->create_attribute("rocm.hip.api", CALI_TYPE_STRING,
                CALI_ATTR_DEFAULT | CALI_ATTR_SKIP_EVENTS);
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

    void handle_sync_api_trace_record(rocmtools_record_tracer_t trec, rocmtools_session_id_t session) {
        if (trec.domain == ACTIVITY_DOMAIN_HIP_API) {
            size_t function_name_size = 0;
            rocmtools_query_hip_tracer_api_data_info_size(
                session, ROCMTOOLS_HIP_FUNCTION_NAME, trec.api_data_handle,
                trec.operation_id, &function_name_size);

            if (function_name_size > 1) {
                char* cptr = nullptr;
                rocmtools_query_hip_tracer_api_data_info(
                    session, ROCMTOOLS_HIP_FUNCTION_NAME, trec.api_data_handle,
                    trec.operation_id, &cptr);

                // skip __hipPush/PopCallConfiguration calls
                if (cptr && strncmp(cptr, "__hip", 5) == 0)
                    return;

                uint64_t duration = trec.timestamps.end.value - trec.timestamps.begin.value;

                const Attribute attr[5] = {
                    m_domain_attr,
                    m_api_attr,
                    m_begin_ts_attr,
                    m_end_ts_attr,
                    m_api_duration_attr
                };
                const Variant data[5] = {
                    Variant(CALI_TYPE_STRING, "hip_api", 7),
                    Variant(CALI_TYPE_STRING, cptr, function_name_size),
                    cali_make_variant_from_uint(trec.timestamps.begin.value),
                    cali_make_variant_from_uint(trec.timestamps.end.value),
                    cali_make_variant_from_uint(duration)
                };

                FixedSizeSnapshotRecord<64> rec;
                Caliper c;

                c.pull_context(m_channel, CALI_SCOPE_THREAD | CALI_SCOPE_PROCESS, rec.builder());

                if (m_correlate_contex) {
                    auto view = rec.view();
                    push_correlation(trec.correlation_id.value,
                        std::vector<Entry> { view.begin(), view.end() });
                }

                c.make_record(5, attr, data, rec.builder());
                m_channel->events().process_snapshot(&c, m_channel, SnapshotView(), rec.view());
            }
        }
    }

    void handle_trace_record(rocmtools_record_tracer_t trec, rocmtools_session_id_t session) {
        std::string activity_name;
        std::string kernel_name;

        if (trec.domain == ACTIVITY_DOMAIN_HIP_OPS) {
            const char* name = reinterpret_cast<const char*>(trec.api_data_handle.handle);

            if (name)
                kernel_name = util::demangle(name);

            char* cptr = nullptr;
            rocmtools_query_hip_tracer_api_data_info(
                    session, ROCMTOOLS_HIP_ACTIVITY_NAME, trec.api_data_handle,
                    trec.operation_id, &cptr);
            if (cptr)
                activity_name = cptr;
        }

        Caliper c;

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

    void flush_records(const rocmtools_record_header_t* record, const rocmtools_record_header_t* end,
            rocmtools_session_id_t session_id, rocmtools_buffer_id_t buffer_id) {
        unsigned count = 0;

        for ( ; record < end; rocmtools_next_record(record, &record, session_id, buffer_id)) {
            switch (record->kind) {
            case ROCMTOOLS_TRACER_RECORD:
            {
                auto* trec = reinterpret_cast<const rocmtools_record_tracer_t*>(record);
                handle_trace_record(*trec, session_id);
                ++count;
                break;
            }
            default:
                break;
            }
        }

        m_num_records += count;
        ++m_num_flushes;

        Log(1).stream() << m_channel->name() << ": rocmtools: flushed " << count << " records\n";
    }

    void init_api_tracing(Channel* channel) {
        auto ret = rocmtools_create_buffer(m_session_id,
                    [](const rocmtools_record_header_t* record, const rocmtools_record_header_t* end_record,
                    rocmtools_session_id_t session_id, rocmtools_buffer_id_t buffer_id) {
                        s_instance->flush_records(record, end_record, session_id, buffer_id);
                    },
                    0x800000, &m_buffer_id);
        if (ret != ROCMTOOLS_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocmtools_create_buffer()", ret);
            return;
        }

        rocmtools_tracer_activity_domain_t filters[2] = {
            ACTIVITY_DOMAIN_HIP_API,
            ACTIVITY_DOMAIN_HIP_OPS
        };

        size_t f_count = 2;
        rocmtools_filter_data_t f_data { filters };

        if (m_trace_api && !m_trace_activities) {
            f_count = 1;
        } else if (!m_trace_api && m_trace_activities) {
            f_data = rocmtools_filter_data_t { &filters[1] };
            f_count = 1;
        } else if (!m_trace_api && !m_trace_activities) {
            f_count = 0;
        }

        rocmtools_filter_id_t api_filter_id;
        ret = rocmtools_create_filter(m_session_id, ROCMTOOLS_API_TRACE,
            f_data, f_count, &api_filter_id, rocmtools_filter_property_t {});
        if (ret != ROCMTOOLS_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocmtools_create_filter()", ret);
            return;
        }

        ret = rocmtools_set_filter_buffer(m_session_id, api_filter_id, m_buffer_id);
        if (ret != ROCMTOOLS_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocmtools_set_filter_buffer()", ret);
            return;
        }

        ret = rocmtools_set_api_trace_sync_callback(m_session_id, api_filter_id,
                [](rocmtools_record_tracer_t rec, rocmtools_session_id_t session){
                    s_instance->handle_sync_api_trace_record(rec, session);
                });
        if (ret != ROCMTOOLS_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocmtools_set_api_trace_sync_callback()", ret);
            return;
        }

        if (m_trace_activities) {
            rocmtools_filter_id_t kernel_filter_id;
            ret = rocmtools_create_filter(m_session_id, ROCMTOOLS_DISPATCH_TIMESTAMPS_COLLECTION,
                    rocmtools_filter_data_t {}, 0, &kernel_filter_id, rocmtools_filter_property_t {});
            if (ret != ROCMTOOLS_STATUS_SUCCESS) {
                print_rocm_error(channel, "rocmtools_create_filter", ret);
            }
            ret = rocmtools_set_filter_buffer(m_session_id, kernel_filter_id, m_buffer_id);
            if (ret != ROCMTOOLS_STATUS_SUCCESS) {
                print_rocm_error(channel, "rocmtools_filter_set_buffer", ret);
            }
        }

        Log(1).stream() << channel->name() << ": rocmtools: API tracing session initialized\n";
    }

    void snapshot_cb(Caliper* c, SnapshotBuilder& rec) {
        rocmtools_timestamp_t timestamp;
        rocmtools_get_timestamp(&timestamp);

        Variant v_now(cali_make_variant_from_uint(timestamp.value));
        Variant v_prev = c->exchange(m_host_timestamp_attr, v_now);

        rec.append(Entry(m_host_duration_attr,
                   Variant(cali_make_variant_from_uint(timestamp.value - v_prev.to_uint()))));
    }

    void post_init_cb(Caliper* c, Channel* channel) {
        auto ret = rocmtools_create_session(ROCMTOOLS_NONE_REPLAY_MODE, &m_session_id);
        if (ret != ROCMTOOLS_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocmtools_create_session()", ret);
            return;
        }

        init_api_tracing(channel);
        ret = rocmtools_start_session(m_session_id);
        if (ret != ROCMTOOLS_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocmtools_start_session()", ret);
            return;
        }

        if (m_record_host_timestamps) {
            rocmtools_timestamp_t timestamp { 0 };
            rocmtools_get_timestamp(&timestamp);

            c->set(m_host_starttime_attr, cali_make_variant_from_uint(timestamp.value));
            c->set(m_host_timestamp_attr, cali_make_variant_from_uint(timestamp.value));

            channel->events().snapshot.connect(
                [](Caliper* c, Channel*, int, SnapshotView, SnapshotBuilder& rec){
                    s_instance->snapshot_cb(c, rec);
                });
        }

        Log(1).stream() << channel->name() << ": rocmtools: session started\n";
    }

    void pre_flush_cb(Caliper*, Channel* channel) {
        auto ret = rocmtools_flush_data(m_session_id, m_buffer_id);
        if (ret != ROCMTOOLS_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocmtools_flush_buffer()", ret);
        }
    }

    void pre_finish_cb(Caliper* c, Channel* channel) {
        auto ret = rocmtools_terminate_session(m_session_id);
        if (ret != ROCMTOOLS_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocmtools_terminate_session()", ret);
        }

        Log(1).stream() << channel->name() << ": rocmtools: session terminated\n";
    }

    void finish_cb(Caliper* c, Channel* channel) {
        rocmtools_finalize();

        Log(1).stream() << channel->name() << ": rocmtools: flushed "
            << m_num_records << " records in " << m_num_flushes << " flushes\n";
    }

    RocmService(Caliper* c, Channel* channel)
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

    static void register_rocmtools(Caliper* c, Channel* channel) {
        auto ret = rocmtools_initialize();
        if (ret != ROCMTOOLS_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocmtools_initialize()", ret);
            return;
        }

        Log(2).stream() << channel->name() << ": rocmtools: using rocmtools version "
            << rocmtools_version_major() << "."
            << rocmtools_version_minor() << "\n";

        s_instance = new RocmService(c, channel);

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

        Log(1).stream() << channel->name() << ": rocmtools service initialized\n";
    }
};

RocmService* RocmService::s_instance = nullptr;

const char* RocmService::s_spec = R"json(
{   "name": "rocmtools",
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

CaliperService rocmtools_service { ::RocmService::s_spec, ::RocmService::register_rocmtools };

}