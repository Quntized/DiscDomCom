#include <ddc/ddc.hpp>
#include <iostream>
#include <Kokkos_Core.hpp>

struct X {};
struct Y {};
struct Z {};

int main() {
    Kokkos::ScopeGuard kokkos_scope;
    
    std::cout << "=== ChunkSpan Constructors and Features ===" << std::endl;
    
    ddc::DiscreteDomain<X, Y, Z> dom(
        ddc::DiscreteElement<X, Y, Z>(0, 0, 0),
        ddc::DiscreteVector<X, Y, Z>(3, 4, 5)
    );
    
    // Create a chunk for testing
    ddc::Chunk<double, ddc::DiscreteDomain<X, Y, Z>> chunk("MyChunk", dom);
    
    // Fill with data
    ddc::for_each(dom, [&](auto elem) {
        auto i = ddc::select<X>(elem).uid();
        auto j = ddc::select<Y>(elem).uid();
        auto k = ddc::select<Z>(elem).uid();
        chunk(elem) = i * 100 + j * 10 + k;
    });
    ddc::ChunkSpan<double, ddc::DiscreteDomain<X, Y, Z>> empty_span;
    std::cout << "  empty_span.data_handle(): " << empty_span.data_handle() << std::endl;
    ddc::ChunkSpan<double, ddc::DiscreteDomain<X, Y, Z>> span1(chunk);
    std::cout << "Created span1 from chunk (lvalue reference)" << std::endl;
    std::cout << "  chunk.data_handle(): " << chunk.data_handle() << std::endl;
    std::cout << "  span1.data_handle(): " << span1.data_handle() << std::endl;
    std::cout << "  Same? " << (chunk.data_handle() == span1.data_handle() ? "YES" : "NO") << std::endl;
    std::cout << "  span1(1,2,3) = " << span1(ddc::DiscreteElement<X,Y,Z>(1,2,3)) << std::endl;
    
    std::cout << "\n=== 3. Copy Constructor ===" << std::endl;
    auto span2 = span1; 
    std::cout << "Created span2 by copying span1" << std::endl;
    std::cout << "  span1.data_handle(): " << span1.data_handle() << std::endl;
    std::cout << "  span2.data_handle(): " << span2.data_handle() << std::endl;
    std::cout << "  Same? " << (span1.data_handle() == span2.data_handle() ? "YES" : "NO") << std::endl;
    
    span2(ddc::DiscreteElement<X,Y,Z>(1,2,3)) = 9999.0;
    std::cout << "  After span2(1,2,3) = 9999:" << std::endl;
    std::cout << "    span1(1,2,3) = " << span1(ddc::DiscreteElement<X,Y,Z>(1,2,3)) << std::endl;
    std::cout << "    chunk(1,2,3) = " << chunk(ddc::DiscreteElement<X,Y,Z>(1,2,3)) << std::endl;
    auto span3 = std::move(span2);  
    std::cout << "  span3.data_handle(): " << span3.data_handle() << std::endl;
    std::cout << "  span3(1,2,3) = " << span3(ddc::DiscreteElement<X,Y,Z>(1,2,3)) << std::endl;
    const auto& const_chunk_ref = chunk;
    ddc::ChunkSpan<const double, ddc::DiscreteDomain<X, Y, Z>> const_span(const_chunk_ref);
    std::cout << "  const_span.data_handle(): " << const_span.data_handle() << std::endl;
    std::cout << "  const_span(1,2,3) = " << const_span(ddc::DiscreteElement<X,Y,Z>(1,2,3)) << std::endl;
    std::cout << "\n=== 7. Copy between ChunkSpans with different element types ===" << std::endl;
    ddc::ChunkSpan<const double, ddc::DiscreteDomain<X, Y, Z>> const_span2(span1);
    std::cout << "  span1.data_handle():       " << span1.data_handle() << std::endl;
    std::cout << "  const_span2.data_handle(): " << const_span2.data_handle() << std::endl;
    std::cout << "  Same? " << (span1.data_handle() == const_span2.data_handle() ? "YES" : "NO") << std::endl;
    
    std::cout << "\n=== 8. Slicing with operator[] ===" << std::endl;
    std::cout << "Creating 2D slice by fixing X=1:" << std::endl;
    auto slice_2d = span1[ddc::DiscreteElement<X>(1)];
    std::cout << "  slice_2d is ChunkSpan<double, DiscreteDomain<Y,Z>>" << std::endl;
    std::cout << "  slice_2d.domain().size() = " << slice_2d.domain().size() << std::endl;
    std::cout << "  slice_2d(2,3) = " << slice_2d(ddc::DiscreteElement<Y,Z>(2,3)) << std::endl;
    
    slice_2d(ddc::DiscreteElement<Y,Z>(2,3)) = 7777.0;
    std::cout << "  After slice_2d(2,3) = 7777:" << std::endl;
    std::cout << "    span1(1,2,3) = " << span1(ddc::DiscreteElement<X,Y,Z>(1,2,3)) << std::endl;
    std::cout << "  Slice modifies original!" << std::endl;
    
    std::cout << "\n=== 9. Creating 1D slice ===" << std::endl;
    auto slice_1d = span1[ddc::DiscreteElement<X,Y>(1,2)];
    std::cout << "Created 1D slice by fixing X=1, Y=2:" << std::endl;
    std::cout << "  slice_1d is ChunkSpan<double, DiscreteDomain<Z>>" << std::endl;
    std::cout << "  slice_1d.domain().size() = " << slice_1d.domain().size() << std::endl;
    std::cout << "  slice_1d(3) = " << slice_1d(ddc::DiscreteElement<Z>(3)) << std::endl;
    
    std::cout << "\n=== 10. Subdomain slicing ===" << std::endl;
    ddc::DiscreteDomain<X, Y, Z> subdomain(
        ddc::DiscreteElement<X, Y, Z>(1, 1, 1),
        ddc::DiscreteVector<X, Y, Z>(2, 2, 2)
    );
    
    auto subspan = span1[subdomain];
    std::cout << "  subspan.domain().size() = " << subspan.domain().size() << " (2×2×2 = 8)" << std::endl;
    std::cout << "  subspan front: " << subspan.domain().front() << std::endl;
    std::cout << "  subspan back:  " << subspan.domain().back() << std::endl;
    
    std::cout << "\n=== 11. Comparing mutable and const spans ===" << std::endl;
    std::cout << "Mutable span (span1):" << std::endl;
    span1(ddc::DiscreteElement<X,Y,Z>(0,0,0)) = 5555.0;
    std::cout << "  Can write: span1(0,0,0) = 5555 ✓" << std::endl;
    std::cout << "  Can read:  span1(0,0,0) = " << span1(ddc::DiscreteElement<X,Y,Z>(0,0,0)) << " ✓" << std::endl;
    
    std::cout << "\nConst span (const_span):" << std::endl;
    std::cout << "  Can read:  const_span(0,0,0) = " << const_span(ddc::DiscreteElement<X,Y,Z>(0,0,0)) << " ✓" << std::endl;
    std::cout << "  Can write: NO ✗ (won't compile)" << std::endl;
    
    std::cout << "\n=== 12. Multiple views to same data ===" << std::endl;
    auto view1 = span1;
    auto view2 = span1;
    auto view3 = view1;
    
    std::cout << "Created 3 views from span1:" << std::endl;
    std::cout << "  view1.data_handle(): " << view1.data_handle() << std::endl;
    std::cout << "  view2.data_handle(): " << view2.data_handle() << std::endl;
    std::cout << "  view3.data_handle(): " << view3.data_handle() << std::endl;
    std::cout << "  All same? " << ((view1.data_handle() == view2.data_handle() && 
                                     view2.data_handle() == view3.data_handle()) ? "YES" : "NO") << std::endl;
    
    view3(ddc::DiscreteElement<X,Y,Z>(2,2,2)) = 3333.0;
    std::cout << "  After view3(2,2,2) = 3333:" << std::endl;
    std::cout << "    view1(2,2,2) = " << view1(ddc::DiscreteElement<X,Y,Z>(2,2,2)) << std::endl;
    std::cout << "    view2(2,2,2) = " << view2(ddc::DiscreteElement<X,Y,Z>(2,2,2)) << std::endl;
    std::cout << "    span1(2,2,2) = " << span1(ddc::DiscreteElement<X,Y,Z>(2,2,2)) << std::endl;
    std::cout << "    chunk(2,2,2) = " << chunk(ddc::DiscreteElement<X,Y,Z>(2,2,2)) << std::endl;
    
    std::cout << "\n=== 13. Passing ChunkSpan to functions ===" << std::endl;
    
    auto modify_data = [](auto span) {
        std::cout << "  Inside function - modifying all values..." << std::endl;
        ddc::for_each(span.domain(), [&](auto elem) {
            span(elem) += 1000.0;
        });
    };
    
    std::cout << "Before: span1(0,0,0) = " << span1(ddc::DiscreteElement<X,Y,Z>(0,0,0)) << std::endl;
    modify_data(span1);
    std::cout << "After:  span1(0,0,0) = " << span1(ddc::DiscreteElement<X,Y,Z>(0,0,0)) << std::endl;
    
    std::cout << "\n=== 14. Using span with STL algorithms ===" << std::endl;
    double* ptr = span1.data_handle();
    auto [min_it, max_it] = std::minmax_element(ptr, ptr + span1.size());
    
    std::cout << "  Min value: " << *min_it << std::endl;
    std::cout << "  Max value: " << *max_it << std::endl;
    std::cout << "  Total elements: " << span1.size() << std::endl;
    

    
    std::cout << "\n=== 16. Assignment operator ===" << std::endl;
    ddc::ChunkSpan<double, ddc::DiscreteDomain<X, Y, Z>> span4;
    span4 = span1;  // Copy assignment
    
    std::cout << "Created span4 and assigned span1 to it:" << std::endl;
    std::cout << "  span1.data_handle(): " << span1.data_handle() << std::endl;
    std::cout << "  span4.data_handle(): " << span4.data_handle() << std::endl;
    std::cout << "  Same? " << (span1.data_handle() == span4.data_handle() ? "YES" : "NO") << std::endl;

    
    return 0;
}

