#include <caliper/CUDATrace.cuh>

#include <caliper/Caliper.h>
#include <caliper/CaliperService.h>
#include <caliper/SnapshotRecord.h>

#include <caliper/common/Log.h>
#include <caliper/common/Node.h>

#include <mutex>
#include <vector>

using namespace cali;
using namespace cali::cudatrace;

namespace
{

class CaliperCUDATrace
{
    struct AttributeInfo {
        Attribute block_attr;
        Attribute duration_attr;
        Attribute timestamp_attr;
        Attribute region_attr;
        Attribute begin_attr;
        Attribute end_attr;
    };

    Node             root_node_;

    AttributeInfo    attributes_;
    std::vector<int> correlation_stack_;
    std::mutex       correlation_stack_mutex_;

    CUDATrace*       tracer_;

    void create_attributes(Caliper* c) {
        attributes_.block_attr = 
            c->create_attribute("cudatrace.block", 
                                CALI_TYPE_UINT, 
                                CALI_ATTR_ASVALUE | CALI_ATTR_SKIP_EVENTS);
        attributes_.duration_attr = 
            c->create_attribute("cudatrace.duration", 
                                CALI_TYPE_UINT, 
                                CALI_ATTR_ASVALUE | CALI_ATTR_SKIP_EVENTS);
        attributes_.timestamp_attr =
            c->create_attribute("cudatrace.timestamp",
                                CALI_TYPE_UINT,
                                CALI_ATTR_ASVALUE | CALI_ATTR_SKIP_EVENTS);
        attributes_.region_attr = 
            c->create_attribute("cudatrace.region",
                                CALI_TYPE_STRING,
                                CALI_ATTR_DEFAULT | CALI_ATTR_SKIP_EVENTS);
        attributes_.begin_attr = 
            c->create_attribute("cudatrace.begin",
                                CALI_TYPE_STRING,
                                CALI_ATTR_DEFAULT | CALI_ATTR_SKIP_EVENTS);
        attributes_.end_attr = 
            c->create_attribute("cudatrace.end",
                                CALI_TYPE_STRING,
                                CALI_ATTR_DEFAULT | CALI_ATTR_SKIP_EVENTS);
    }

    void flush_cb(Caliper* c, Channel* channel, SnapshotFlushFn snap_fn) {
        std::vector<uint64_t> last_block_timestamp(8192, 0);
        std::vector<Node*> region_stack(8192, &root_node_);

        unsigned count = tracer_->flush([this,&last_block_timestamp,&region_stack,c,snap_fn](const CUDATrace::region_entry_t& r, const CUDATrace::trace_entry_t& e){
                std::vector<Entry> rec;
                rec.reserve(5);

                auto b = e.block_id;

                if (e.correlation_id >= 0)
                    rec.push_back(Entry(c->node(e.correlation_id)));

                Node* node = &root_node_;

                if (e.event == CUDATrace::event_t::BEGIN) {
                    if (b < region_stack.size()) {
                        node = region_stack[b];
                        region_stack[b] = c->make_tree_entry(attributes_.region_attr, Variant(r.name), node);
                    }
                    rec.push_back(Entry(c->make_tree_entry(attributes_.begin_attr, Variant(r.name), node)));
                } else {
                    if (b < region_stack.size()) {
                        node = region_stack[b];
                        if (node != &root_node_)
                            region_stack[b] = node->parent();
                    }
                    rec.push_back(Entry(c->make_tree_entry(attributes_.end_attr, Variant(r.name), node)));
                }

                if (b < last_block_timestamp.size()) {
                    if (last_block_timestamp[b] > 0) {
                        auto duration = e.timestamp - last_block_timestamp[b];
                        rec.push_back(Entry(attributes_.duration_attr, duration));
                    }
                    last_block_timestamp[b] = e.timestamp;
                }

                rec.push_back(Entry(attributes_.block_attr, cali_make_variant_from_uint(b)));
                rec.push_back(Entry(attributes_.timestamp_attr, cali_make_variant_from_uint(e.timestamp)));

                snap_fn(*c, rec);
            });

        tracer_->clear();

        Log(1).stream() << channel->name() << ": cudatrace: flushed " << count << " records." << std::endl;
    }

    void post_begin_cb(Caliper* c, const Attribute& attr) {
        if (!attr.is_nested())
            return;
        
        const Node* node = c->get(attr).node();
        int id = node ? static_cast<int>(node->id()) : -1;

        if (id >= 0) {
            tracer_->set_correlation(node->id());
    
            std::lock_guard<std::mutex>
                g(correlation_stack_mutex_);

            correlation_stack_.push_back(id);
        }
    }

    void post_end_cb(const Attribute& attr) {
        if (!attr.is_nested())
            return;
        
        int id = -1;

        {
            std::lock_guard<std::mutex>
                g(correlation_stack_mutex_);

            if (!correlation_stack_.empty()) {
                id = correlation_stack_.back();
                correlation_stack_.pop_back();
            }
        }

        tracer_->set_correlation(id);
    }

    CaliperCUDATrace(Caliper* c, CUDATrace* tracer)
        : root_node_ { CALI_INV_ID, CALI_INV_ID, Variant() },
          tracer_ { tracer }
    { 
        create_attributes(c);
    }

    ~CaliperCUDATrace()
    { }

public:

    static void register_cudatrace(Caliper* c, Channel* channel) {
        CUDATrace* tracer = cudatrace_init(2*1024*1024, 8192);

        CaliperCUDATrace* instance = new CaliperCUDATrace(c, tracer);

        channel->events().post_begin_evt.connect(
            [instance](Caliper* c, Channel*, const Attribute& attr, const Variant&){
                    instance->post_begin_cb(c, attr);
                });
        channel->events().post_end_evt.connect(
            [instance](Caliper*, Channel*, const Attribute& attr, const Variant&){
                    instance->post_end_cb(attr);
                });
        channel->events().flush_evt.connect(
            [instance](Caliper* c, Channel* channel, const SnapshotView&, SnapshotFlushFn fn){
                    instance->flush_cb(c, channel, fn);
                });
        channel->events().finish_evt.connect(
            [instance](Caliper* c, Channel* channel){
                    delete instance;
                    cudatrace_finalize();
                });

        Log(1).stream() << channel->name() << ": registered cudatrace service" << std::endl;
    }
};

} // namespace [anonymous]

namespace cali
{

CaliperService cudatrace_service { "cudatrace", CaliperCUDATrace::register_cudatrace };

}
