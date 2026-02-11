#include <ddc/ddc.hpp>
#include <iostream>
#include <Kokkos_Core.hpp>

struct X {};
struct Y {};
struct Z {};

int main() {
    Kokkos::initialize();
    {
      
        ddc::DiscreteDomain<X, Y, Z> dom(
            ddc::DiscreteElement<X, Y, Z>(0, 0, 0), 
            ddc::DiscreteVector<X, Y, Z>(5, 4, 3)
        );
        
        ddc::Chunk<double, ddc::DiscreteDomain<X, Y, Z>> data("MyData", dom);
        
        ddc::for_each(dom, [&](auto elem) {
            auto i = ddc::select<X>(elem).uid();
            auto j = ddc::select<Y>(elem).uid();
            auto k = ddc::select<Z>(elem).uid();
            data(elem) = i + j + k;
        });

        
        auto kokkos_view = data.allocation_kokkos_view();
        std::cout << "Kokkos::View properties: " << std::endl;
        std::cout << "  Rank = " << kokkos_view.rank() << std::endl;
        std::cout << "  extent(0) = " << kokkos_view.extent(0) << std::endl;
        std::cout << "  extent(1) = " << kokkos_view.extent(1) << std::endl;
        std::cout << "  extent(2) = " << kokkos_view.extent(2) << std::endl;
        std::cout << "  size      = " << kokkos_view.size() << std::endl;

        std::cout << "View at (1,2,1) = " << kokkos_view(1, 2, 1) << std::endl;
        kokkos_view(1, 2, 1) = 121.0;
        std::cout << "View at (1,2,1) after modification = " << kokkos_view(1, 2, 1) << std::endl;
        std::cout << "Data(1,2,1) = " << data(ddc::DiscreteElement<X, Y, Z>(1, 2, 1)) << std::endl;

     
        Kokkos::parallel_for(
            "Scaled data", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0, 0, 0}, 
                {(int)kokkos_view.extent(0), (int)kokkos_view.extent(1), (int)kokkos_view.extent(2)}
            ), 
            KOKKOS_LAMBDA(int i, int j, int k) {
                kokkos_view(i, j, k) *= 2.0;
            }
        );
        Kokkos::fence();

        std::cout << "After Scaling *2.0" << std::endl;
        std::cout << "  Data(0,0,0) = " << data(dom.front()) << std::endl;
        std::cout << "  Data(1,2,1) = " << data(ddc::DiscreteElement<X, Y, Z>(1, 2, 1)) << std::endl;

     
        const auto& const_data = data;
        auto const_view = const_data.allocation_kokkos_view();
        std::cout << "Constant kokkos view at 1,1,1 = " << const_view(1, 1, 1) << std::endl;

        auto span = data.span_view();
        std::cout << "span view properties: " << std::endl;
        std::cout << "  Domain size " << span.domain().size() << std::endl;

        auto elem = ddc::DiscreteElement<X, Y, Z>(3, 2, 1);
        std::cout << "data(3,2,1) = " << data(elem) << std::endl;
        std::cout << "span(3,2,1) = " << span(elem) << std::endl;

        span(elem) = 2872.3;
        std::cout << "After modification of data's span = " << data(elem) << std::endl;

        auto cspan = data.span_cview();
        std::cout << "Constant span view at 3,2,1 = " << cspan(elem) << std::endl;

        auto process_span = [&](auto spn) {
            double sum = 0.0;
            ddc::for_each(spn.domain(), [&](auto ee) {
                sum += spn(ee);
            });
            return sum;
        };

        double total = process_span(data.span_cview());
        std::cout << "sum = " << total << std::endl;

        
        auto slice = data[ddc::DiscreteElement<X>(2)];
        auto slice_span = slice.span_view();
        std::cout << "Slice span domain size " << slice_span.domain().size() << std::endl;

        ddc::for_each(slice_span.domain(), [&](auto elem2d) {
            slice_span(elem2d) = 555.0;
        });

        std::cout << "After modifying slice via span:" << std::endl;
        std::cout << "  data(2,0,0) = " << data(ddc::DiscreteElement<X, Y, Z>(2, 0, 0)) << std::endl;


        std::cout << "\n=== EXAMPLE 9: Template deduction guides ===" << std::endl;
        ddc::KokkosAllocator<float, Kokkos::HostSpace> float_alloc;
        ddc::Chunk float_chunk("FloatData", dom, float_alloc);

        std::cout << "Float chunk created with deduced type" << std::endl;
        std::cout << "  Size: " << float_chunk.domain().size() << std::endl;

       
        std::cout << "\n=== EXAMPLE 10: Kokkos reduction ===" << std::endl;
        auto reduction_view = data.allocation_kokkos_view();
        double max_val = 0.0;

        Kokkos::parallel_reduce(
            "find_max",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0, 0, 0},
                {(int)reduction_view.extent(0), (int)reduction_view.extent(1), (int)reduction_view.extent(2)}
            ),
            KOKKOS_LAMBDA(int i, int j, int k, double& lmax) {
                if (reduction_view(i, j, k) > lmax) {
                    lmax = reduction_view(i, j, k);
                }
            },
            Kokkos::Max<double>(max_val)
        );
        Kokkos::fence();
        std::cout << "Maximum value found: " << max_val << std::endl;

    
        std::cout << "\n=== EXAMPLE 11: Zero-copy interop ===" << std::endl;
        auto shared_view = data.allocation_kokkos_view();

        std::cout << "Memory addresses (should be identical):" << std::endl;
        std::cout << "  DDC data_handle(): " << data.data_handle() << std::endl;
        std::cout << "  Kokkos view data(): " << shared_view.data() << std::endl;
        std::cout << "  Same memory: " << (data.data_handle() == shared_view.data() ? "YES" : "NO") << std::endl;

       
        std::cout << "\n=== EXAMPLE 12: Working with subdomains ===" << std::endl;
        ddc::DiscreteDomain<X, Y, Z> subdomain(
            ddc::DiscreteElement<X, Y, Z>(1, 1, 0),
            ddc::DiscreteVector<X, Y, Z>(3, 2, 2)
        );

        auto subchunk = data[subdomain];
        auto sub_span = subchunk.span_view();

        std::cout << "Subdomain span size: " << sub_span.domain().size() << std::endl;
        std::cout << "  data(1,1,0) = " << data(ddc::DiscreteElement<X, Y, Z>(1, 1, 0)) << std::endl;

        ddc::for_each(sub_span.domain(), [&](auto sub_elem) {
            sub_span(sub_elem) += 1000.0;
        });

        std::cout << "After adding 1000 to subdomain:" << std::endl;
        std::cout << "  data(1,1,0) = " << data(ddc::DiscreteElement<X, Y, Z>(1, 1, 0)) << std::endl;
    }
    Kokkos::finalize();

    return 0;
}
