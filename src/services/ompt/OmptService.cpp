// Copyright (c) 2019, Lawrence Livermore National Security, LLC.  
// See top-level LICENSE file for details.

#include "caliper/Caliper.h"

#include "caliper/CaliperService.h"

#include "caliper/common/Log.h"

#include "../util/ChannelList.hpp"

#include <ompt.h>

using namespace cali;
using util::ChannelList;

namespace 
{

struct OmptAPI {
    ompt_set_callback_t      set_callback     { nullptr };
    ompt_get_state_t         get_state        { nullptr };
    ompt_enumerate_states_t  enumerate_states { nullptr };

    // std::vector<std::string> states;

    bool
    init(ompt_function_lookup_t lookup) {
        set_callback     = (ompt_set_callback_t)     (*lookup)("ompt_set_callback");
        get_state        = (ompt_get_state_t)        (*lookup)("ompt_get_state");
        enumerate_states = (ompt_enumerate_states_t) (*lookup)("ompt_enumerate_states");

        if (!set_callback || !get_state || !enumerate_states)
            return false;

        // enumerate states

        // states.reserve(32);

        // int state = ompt_state_undefined;
        // const char* state_name;

        // while ((*api.enumerate_states)(state, &state, &state_name)) {
        //     if (states.size() <= state)
        //         states.resize(state);

        //     states[state] = std::string(state_name);
        // }

        return true;
    }
} api;

ChannelList* all_channels       { nullptr };
ChannelList* callback_channels  { nullptr };

Attribute    region_attr        { Attribute::invalid };
Attribute    thread_type_attr   { Attribute::invalid };
Attribute    state_attr         { Attribute::invalid };

// 
// --- The OMPT callbacks
//

void cb_thread_begin(ompt_thread_t type, ompt_data_t*)
{
    for (ChannelList* ptr = all_channels; ptr; ptr = ptr->next)
        if (ptr->channel->is_active()) {
            Caliper c;

            switch (type) {
            case ompt_thread_initial:  
                c.begin(ptr->channel, thread_type_attr, Variant("initial"));
                break;
            case ompt_thread_worker:
                c.begin(ptr->channel, thread_type_attr, Variant("worker"));
                break;
            case ompt_thread_other:
                c.begin(ptr->channel, thread_type_attr, Variant("other"));
                break;
            default:
                c.begin(ptr->channel, thread_type_attr, Variant("unknown"));
            }
        }
}

void cb_thread_end(ompt_data_t*)
{
    for (ChannelList* ptr = all_channels; ptr; ptr = ptr->next)
        if (ptr->channel->is_active()) {
            Caliper c;

            if (!c.get(ptr->channel, thread_type_attr).is_empty())
                c.end(ptr->channel, thread_type_attr);
        }
}

void cb_parallel_begin(ompt_data_t*, ompt_frame_t*, ompt_data_t*, unsigned, int, const void*)
{
    for (ChannelList* ptr = callback_channels; ptr; ptr = ptr->next)
        if (ptr->channel->is_active()) {
            Caliper c;
            c.begin(ptr->channel, region_attr, Variant("parallel"));
        }
}

void cb_parallel_end(ompt_data_t*, ompt_data_t*, int, const void*)
{
    for (ChannelList* ptr = callback_channels; ptr; ptr = ptr->next)
        if (ptr->channel->is_active()) {
            Caliper c;
            c.end(ptr->channel, region_attr);
        }
}


//
// --- OMPT management
//


void setup_callbacks()
{
    const struct callback_info {
        ompt_callbacks_t cb;
        ompt_callback_t  fn;
    } callbacks[] = {
        { ompt_callback_thread_begin,   reinterpret_cast<ompt_callback_t>(cb_thread_begin)   },
        { ompt_callback_thread_end,     reinterpret_cast<ompt_callback_t>(cb_thread_end)     },
        { ompt_callback_parallel_begin, reinterpret_cast<ompt_callback_t>(cb_parallel_begin) },
        { ompt_callback_parallel_end,   reinterpret_cast<ompt_callback_t>(cb_parallel_end)   }
    };

    for (auto info : callbacks)
        (*(api.set_callback))(info.cb, info.fn);
}

int initialize_ompt(ompt_function_lookup_t lookup, int initial_device_num, ompt_data_t* tool_data) 
{
    if (!api.init(lookup)) {
        Log(0).stream() << "Cannot initialize OMPT API" << std::endl;
        return 0;
    }

    setup_callbacks();

    return 42;
}

void finalize_ompt(ompt_data_t* tool_data) 
{

}

ompt_start_tool_result_t start_tool_result { initialize_ompt, finalize_ompt, { 0 } };


//
// --- Caliper management
//

void create_attributes(Caliper* c)
{
    region_attr = 
        c->create_attribute("ompt.region", CALI_TYPE_STRING, CALI_ATTR_SCOPE_THREAD);

    thread_type_attr = 
        c->create_attribute("ompt.thread.type", CALI_TYPE_STRING, CALI_ATTR_SCOPE_THREAD);

    state_attr =
        c->create_attribute("ompt.state", CALI_TYPE_STRING, CALI_ATTR_SCOPE_THREAD);
}

void register_ompt_service(Caliper* c, Channel* chn)
{
    create_attributes(c);

    ChannelList::add(&all_channels, chn);
    ChannelList::add(&callback_channels, chn);

    chn->events().finish_evt.connect(
        [](Caliper* c, Channel* chn){
            ChannelList::remove(&callback_channels, chn);
            ChannelList::remove(&all_channels, chn);
        });

    Log(1).stream() << chn->name() << ": " << "Registered OMPT service" << std::endl;
}

} // namespace [anonymous]

extern "C" {

ompt_start_tool_result_t*
ompt_start_tool(unsigned omp_version, const char* runtime_version) {
    Log(2).stream() << "OMPT is available. Using " << runtime_version << std::endl;

    return &::start_tool_result;
}

}

namespace cali
{

CaliperService ompt_service { "ompt", ::register_ompt_service };

}
