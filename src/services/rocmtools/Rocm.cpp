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

#include <cassert>

using namespace cali;

namespace
{

void print_rocm_error(Channel* channel, const char* msg, rocmtools_status_t s)
{
    Log(0).stream() << channel->name() << ": " << msg << " error: "
        << rocmtools_error_string(s)
        << "\n";
}

class RocmService
{
    Attribute m_api_attr;
    Attribute m_begin_ts_attr;
    Attribute m_end_ts_attr;
    Attribute m_duration_attr;
    Attribute m_host_timestamp_attr;
    Attribute m_host_duration_attr;

    Channel*  m_channel;

    rocmtools_session_id_t m_session_id;
    rocmtools_buffer_id_t  m_buffer_id;

    static RocmService* s_instance;

    void create_attributes(Caliper* c) {
        m_api_attr =
            c->create_attribute("rocm.hip.api", CALI_TYPE_STRING,
                CALI_ATTR_DEFAULT | CALI_ATTR_SKIP_EVENTS);
        m_begin_ts_attr =
            c->create_attribute("rocm.starttime", CALI_TYPE_UINT,
                CALI_ATTR_ASVALUE | CALI_ATTR_SKIP_EVENTS);
        m_end_ts_attr =
            c->create_attribute("rocm.endtime", CALI_TYPE_UINT,
                CALI_ATTR_ASVALUE | CALI_ATTR_SKIP_EVENTS);
        m_duration_attr =
            c->create_attribute("rocm.activity.duration", CALI_TYPE_UINT,
                CALI_ATTR_ASVALUE | CALI_ATTR_SKIP_EVENTS | CALI_ATTR_AGGREGATABLE);

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
    }

    void handle_trace_record(rocmtools_record_tracer_t record, rocmtools_session_id_t session) {
        if (record.domain == ACTIVITY_DOMAIN_HIP_API) {
            size_t function_name_size = 0;
            rocmtools_query_hip_tracer_api_data_info_size(
                session, ROCMTOOLS_HIP_FUNCTION_NAME, record.api_data_handle,
                record.operation_id, &function_name_size);

            if (function_name_size > 1) {
                char* cptr = nullptr;
                rocmtools_query_hip_tracer_api_data_info(
                    session, ROCMTOOLS_HIP_FUNCTION_NAME, record.api_data_handle,
                    record.operation_id, &cptr);

                FixedSizeSnapshotRecord<4> rec;
                Caliper c;

                uint64_t duration = record.timestamps.end.value - record.timestamps.begin.value;

                const Attribute attr[4] = {
                    m_api_attr,
                    m_begin_ts_attr,
                    m_end_ts_attr,
                    m_duration_attr
                };
                const Variant data[4] = {
                    Variant(CALI_TYPE_STRING, cptr, function_name_size),
                    cali_make_variant_from_uint(record.timestamps.begin.value),
                    cali_make_variant_from_uint(record.timestamps.end.value),
                    cali_make_variant_from_uint(duration)
                };

                c.make_record(4, attr, data, rec.builder());
                c.push_snapshot(m_channel, rec.view());
            }
        }
    }

    void flush_records(const rocmtools_record_header_t* record, const rocmtools_record_header_t* end,
            rocmtools_session_id_t session_id, rocmtools_buffer_id_t buffer_id) {
        unsigned count = 0;

        while (record < end) {
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

        Log(1).stream() << m_channel->name() << ": rocmtools: flushed " << count << " records\n";
    }

    void init_api_tracing(Channel* channel) {
        auto ret = rocmtools_create_buffer(m_session_id,
                    [](const rocmtools_record_header_t* record, const rocmtools_record_header_t* end_record,
                    rocmtools_session_id_t session_id, rocmtools_buffer_id_t buffer_id) {
                        s_instance->flush_records(record, end_record, session_id, buffer_id);
                    },
                    0x0, &m_buffer_id);
        if (ret != ROCMTOOLS_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocmtools_create_buffer()", ret);
            return;
        }

        rocmtools_tracer_activity_domain_t filter = ACTIVITY_DOMAIN_HIP_API;

        rocmtools_filter_id_t api_filter_id;
        ret = rocmtools_create_filter(m_session_id, ROCMTOOLS_API_TRACE,
            rocmtools_filter_data_t { &filter }, 1,
            &api_filter_id, rocmtools_filter_property_t {});
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
                    s_instance->handle_trace_record(rec, session);
                });
        if (ret != ROCMTOOLS_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocmtools_set_api_trace_sync_callback()", ret);
            return;
        }

        Log(1).stream() << channel->name() << ": rocmtools API tracing session initialized\n";
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

        channel->events().snapshot.connect(
            [](Caliper* c, Channel*, int, SnapshotView, SnapshotBuilder& rec){
                s_instance->snapshot_cb(c, rec);
            });

        Log(1).stream() << channel->name() << ": rocmtools session started\n";
    }

    void pre_finish_cb(Caliper* c, Channel* channel) {
        auto ret = rocmtools_terminate_session(m_session_id);
        if (ret != ROCMTOOLS_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocmtools_terminate_session()", ret);
        }

        ret = rocmtools_flush_data(m_session_id, m_buffer_id);
        if (ret != ROCMTOOLS_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocmtools_flush_buffer()", ret);
        }

        Log(1).stream() << channel->name() << ": rocmtools session terminated\n";
    }

    void finish_cb(Caliper* c, Channel* channel) {
        rocmtools_finalize();
    }

    RocmService(Caliper* c, Channel* channel)
        : m_channel { channel }
    {
        create_attributes(c);
    }

public:

    static const char* s_spec;

    static void register_rocmtools(Caliper* c, Channel* channel) {
        auto ret = rocmtools_initialize();
        if (ret != ROCMTOOLS_STATUS_SUCCESS) {
            print_rocm_error(channel, "rocmtools_initialize()", ret);
            return;
        }

        s_instance = new RocmService(c, channel);

        channel->events().post_init_evt.connect(
            [](Caliper* c, Channel* channel){
                s_instance->post_init_cb(c, channel);
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
        {   "name": "record_kernel_names",
            "type": "bool",
            "description": "Record kernel names when activity tracing is enabled",
            "value": "false"
        },
        {   "name": "snapshot_duration",
            "type": "bool",
            "description": "Record duration of host-side activities using ROCm timestamps",
            "value": "false"
        },
        {   "name": "snapshot_timestamps",
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