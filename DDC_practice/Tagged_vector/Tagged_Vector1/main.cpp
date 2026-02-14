#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>

#include <ddc/ddc.hpp>
#include <Kokkos_Core.hpp>

//continuous dimension tag;
struct X;
struct Y;
struct DDimX : ddc::UniformPointSampling<X> {};
struct DDimY : ddc::UniformPointSampling<Y> {};

template <class ChunkType>
void print_2d(std::string const& label, ChunkType const& chunk)
{
  std::cout<<"  "<<label<<"\n";
  auto dom = chunk.domain();
  ddc::for_each(ddc::select<DDimX>(dom), [&](ddc::DiscreteElement<DDimX> ix){
    std::cout<<"  ";
    ddc::for_each(ddc::select<DDimY>(dom), [&](ddc::DiscreteElement<DDimY> iy){
      std::cout<<std::setw(8)<<std::fixed<<std::setprecision(2)<<chunk(ix,iy)<<"  ";
    });
    std::cout<<"\n";
  });
}
int main(int argc, char** argv){
  ddc::ScopeGuard const ddc_scope(argc,argv);
  {
  ddc::Coordinate<X> x_coord(3.14);
  ddc::Coordinate<X,Y> point(1.0,2.0);
  std::cout<<"  point  =( "<<ddc::get<X>(point)<<" , "<<ddc::get<Y>(point)<<" ) \n";
  ddc::Coordinate<X,Y> offset(0.5,0.4);
  ddc::Coordinate<X,Y> moved = point + offset;
  std::cout<<"  point+offset = ( "<<ddc::get<X>(moved)<<" , "<<ddc::get<Y>(moved)<<" ) \n";
  ddc::Coordinate<X> just_x = ddc::select<X>(point);
  std::cout<<"   select<X>(point)  =  "<<just_x<<" \n";
  }
  std::cout<<" \n"<<std::endl;
  {
    ddc::DiscreteVector<DDimX> dvx(5);
    ddc::DiscreteVector<DDimY> dvy(10);
    ddc::DiscreteVector<DDimX,DDimY> dv2d(3,7);
    std::cout<<" dvx = "<<ddc::get<DDimX>(dvx)<<std::endl;
    std::cout<<" dvy = "<<ddc::get<DDimY>(dvy)<<std::endl;
    std::cout<<" dvx2d = "<<ddc::get<DDimX>(dv2d)<<std::endl;
    std::cout<<" dvy2d = "<<ddc::get<DDimY>(dv2d)<<std::endl;
    ddc::Coordinate<DDimX,DDimY> dv_1(1,2);
    ddc::Coordinate<DDimX,DDimY> dv_2(2,3);
    ddc::Coordinate<DDimX,DDimY> dv = dv_1+dv_2;
    std::cout<<" dv_1+dv_2 = ( "<<ddc::get<DDimX>(dv)<< " , "<<ddc::get<DDimY>(dv)<<" ) . "<<std::endl;
    auto dv_n = -dv_1;
    std::cout<<"-dv_1 = ( "<<ddc::get<DDimX>(dv_n)<<" , "<<ddc::get<DDimY>(dv_n)<<" ) . "<<std::endl;
    ddc::DiscreteVector<DDimX> just_x = ddc::select<DDimX>(dv2d);
    std::cout<<" select<DDimX>(dv2d) =  "<<just_x<<"  "<<std::endl;
  }
  {
    ddc::DiscreteElement<DDimX> ix(3);
    ddc::DiscreteElement<DDimY> iy(5);
    ddc::DiscreteElement<DDimX,DDimY> ixy(ix,iy);
    //std::cout<<" ix = "<<ddc::get<DDimX>(ix).uid()<<std::endl;
    //std::cout<<" iy = "<<ddc::get<DDimY>(iy).uid()<<std::endl;
    std::cout<<" ix = "<<ix<<std::endl;
    std::cout<<" iy = "<<iy<<std::endl;
    //std::cout<<" ix = "<<ddc::get<DDimX>(ixy)<<std::endl;
    ddc::DiscreteVector<DDimX> step_x(2);
    ddc::DiscreteVector<DDimY> step_y(4);
    ddc::DiscreteElement<DDimX> ix_moved = ix+step_x;
    std::cout<<" ix+step_x(2)  =  "<<ix_moved<<std::endl;
    ddc::DiscreteElement<DDimX> ix1(10);
    ddc::DiscreteVector<DDimX> dist = ix1 - ix;
    std::cout<<" ix1 - ix = "<<dist<<std::endl;
    auto ix_next = ix+1;
    auto ix_prev = ix - 1;
    std::cout<<" ix_next = "<<ix_next<<std::endl;
    std::cout<<" ix_prev = "<<ix_prev<<std::endl;
    ddc::DiscreteElement<DDimX> proj_x = ddc::select<DDimX>(ixy);
    std::cout<<" projection_x = "<<proj_x<<std::endl;
    ddc::DiscreteElement<DDimY,DDimX> iyx(iy,ix);
    //ddc::DiscreteElement<DDimX,DDimY> ixy(iyx);
    std::cout<<"  Reordered = (Y,X)  ----> (X,Y)  "<<"  (Y,X) ---> ( "<<ddc::select<DDimY>(iyx)<<" , "<<ddc::select<DDimX>(iyx)<<" ) . (X,Y) ----> ( "<<ddc::select<DDimX>(ixy)<<" , "<<ddc::select<DDimY>(ixy)<<" ). "<<std::endl;
  }
  {
    ddc::DiscreteElement<DDimX> const ix_start(0);
    ddc::DiscreteVector<DDimX> const nx(8);
    ddc::DiscreteElement<DDimY> const iy_start(0);
    ddc::DiscreteVector<DDimY> const ny(5);
    ddc::DiscreteDomain<DDimX> dom_x(ix_start,nx);
    ddc::DiscreteDomain<DDimY> dom_y(iy_start,ny);
    std::cout<<"x_domain.front() ------>  "<<ddc::select<DDimX>(dom_x.front())<<",domain_x size ------->  "<<dom_x.size()<<std::endl;
    //std::cout<<"domain.front() ------>  "<<ddc::select<DDimX>(dom_x.front()+1)<<", size ------->  "<<dom_x.size()<<std::endl; <----- works
    std::cout<<"y_domain.front() ------>  "<<ddc::select<DDimY>(dom_y.front())<<", domain_y size ------->  "<<dom_y.size()<<std::endl;
    ddc::DiscreteDomain<DDimX,DDimY> dom(dom_x,dom_y);
    std::cout<<" domain.front() ----> ( "<<ddc::select<DDimX>(dom.front())<<" , "<<ddc::select<DDimY>(dom.front())<<" ) . "<< ". EXTENTS = ( "<<dom.extent<DDimX>()<<" , "<<dom.extent<DDimY>()<<" ) . "<<"\n";
    // ddc::select<DDimX>(dom_xy.extents())  also would work.
    ddc::DiscreteDomain<DDimX> proj_x = ddc::select<DDimX>(dom);
    std::cout<<" ddc::select<DDimX>(proj_x) -----> "<<proj_x.size()<<std::endl;
    std::cout<<"  First 3 elements of dom_x ----------------------------------------------------------------------------> "<<std::endl;
    int count = 0;
    ddc::for_each(dom_x,[&](ddc::DiscreteElement<DDimX> ix){
      if(count++<3){
        std::cout<<ddc::select<DDimX>(ix)<<" ";
      }
    });
    std::cout<<" \n";
  }
  std::cout<<" \n";
  {
    ddc::DiscreteDomain<DDimX> dom_x(ddc::DiscreteElement<DDimX>(0),ddc::DiscreteVector<DDimX>(10));
    ddc::DiscreteDomain<DDimY> dom_y(ddc::DiscreteElement<DDimY>(0), ddc::DiscreteVector<DDimY>(8));
    ddc::DiscreteDomain<DDimX,DDimY> domain(dom_x,dom_y);
    ddc::Chunk temperature("temperature",domain,ddc::DeviceAllocator<double>());
    std::cout<<" Chunk temprature label = "<<temperature.label()<<"\n";
    std::cout<<" Chunk size = "<<temperature.domain().size()<<"\n";
    ddc::for_each(domain, [&](ddc::DiscreteElement<DDimX,DDimY> ixy){
      ddc::DiscreteElement<DDimX> ix(ixy);
      ddc::DiscreteElement<DDimY> iy(ixy);
      temperature(ix,iy) = ddc::select<DDimX>(ix).uid() + ddc::select<DDimY>(iy).uid();
    });
  print_2d("temperature after fill", temperature);
  ddc::Chunk temp_moved(std::move(temperature));
  std::cout<<" After move: size   =   "<<temp_moved.domain().size()<<std::endl;
  
  }
  std::cout<<" \n"<<std::endl;
  ddc::DiscreteDomain<DDimX> dom_x(ddc::DiscreteElement<DDimX>(0),ddc::DiscreteVector<DDimX>(10));
  ddc::DiscreteDomain<DDimY> dom_y(ddc::DiscreteElement<DDimY>(0), ddc::DiscreteVector<DDimY>(8));
  ddc::DiscreteDomain<DDimX,DDimY> dom(dom_x,dom_y);
  ddc::Chunk data("data",dom,ddc::DeviceAllocator<double>());
  ddc::ChunkSpan span = data.span_view();
  ddc::ChunkSpan cspan = data.span_cview();
  ddc::for_each(dom, [&](ddc::DiscreteElement<DDimX,DDimY> ixy){
    ddc::DiscreteElement<DDimX> ix(ixy);
    ddc::DiscreteElement<DDimY> iy(ixy);
    span(ix,iy) = ddc::select<DDimX>(ix).uid() + ddc::select<DDimY>(iy).uid();
  });
  print_2d("span ", span);
  std::cout<<" span.domain().size() -----> "<<span.domain().size()<<std::endl;
  std::cout<<"const_span(1,2,1) ----> "<<cspan(ddc::DiscreteElement<DDimX,DDimY>(1,2))<<std::endl;
  std::cout<<"X extent od span ------> "<<span.extent<DDimX>()<<std::endl;
  std::cout<<"Y extent od span ------> "<<span.extent<DDimY>()<<std::endl;
  {
    ddc::DiscreteDomain<DDimX> dom_x(ddc::DiscreteElement<DDimX>(0),ddc::DiscreteVector<DDimX>(10));
    ddc::DiscreteDomain<DDimY> dom_y(ddc::DiscreteElement<DDimY>(0), ddc::DiscreteVector<DDimY>(8));
    ddc::DiscreteDomain<DDimX,DDimY> dom(dom_x,dom_y);
    ddc::Chunk grid("grid", dom, ddc::DeviceAllocator<double>());
    ddc::ChunkSpan span = grid.span_view();
    ddc::for_each(span.domain(), [&](ddc::DiscreteElement<DDimX,DDimY> ixy){
      ddc::DiscreteElement<DDimX> ix(ixy);
      ddc::DiscreteElement<DDimY> iy(ixy);
      span(ix,iy) = 10*ddc::select<DDimX>(ix).uid() + ddc::select<DDimY>(iy).uid();
  });
    print_2d("10x8 matrix data = ",span);
    auto row_x2 = span[ddc::DiscreteElement<DDimX>(2)];
    std::cout<<" row x=2 slice ------> "<<std::endl;
    ddc::for_each(row_x2.domain(),[&](auto elem){
      std::cout<<row_x2(elem)<<"    ";
    });
    std::cout<<"\n";
    ddc::DiscreteDomain<DDimX> sub_x(ddc::DiscreteElement<DDimX>(1),ddc::DiscreteVector<DDimX>(3));
    auto subdom_x = span[sub_x];
    print_2d("Subgrid X = 1 to X = 3 ---> ",subdom_x);
    ddc::DiscreteDomain<DDimY> sub_y(ddc::DiscreteElement<DDimY>(1),ddc::DiscreteVector<DDimY>(3));
    auto subdom_y = span[sub_y];
    print_2d(" subgrid Y = 1 to Y = 3",subdom_y);
    ddc::DiscreteElement<DDimX,DDimY> sec_xy(ddc::DiscreteElement<DDimX>(1),ddc::DiscreteElement<DDimY>(3));
    auto dissect = span(sec_xy);
    std::cout<<" sub_grid(X=1,Y=3) ------> "<<dissect<<std::endl;;
  }
  {
    ddc::DiscreteDomain<DDimX, DDimY> dom(ddc::DiscreteDomain<DDimX>(ddc::DiscreteElement<DDimX>(0), ddc::DiscreteVector<DDimX>(3)),ddc::DiscreteDomain<DDimY>(ddc::DiscreteElement<DDimY>(0), ddc::DiscreteVector<DDimY>(3)));
    ddc::Chunk data("order_test", dom, ddc::DeviceAllocator<double>());
    ddc::ChunkSpan span = data.span_view();
    ddc::DiscreteElement<DDimX> ix(1);
    ddc::DiscreteElement<DDimY> iy(2);
    span(ix, iy) = 42.0;
    std::cout << "  span(ix=1, iy=2) = " << span(ix, iy) << "\n";
    std::cout << "  span(iy=2, ix=1) = " << span(iy, ix) << " (same!)\n";
    ddc::DiscreteElement<DDimY, DDimX> iyx(iy, ix);
    std::cout << "  span(iyx{Y=2,X=1}) = " << span(iyx) << " (same!)\n";
    print_2d("",span);
  }
  {
  ddc::DiscreteDomain<DDimX, DDimY> dom(ddc::DiscreteDomain<DDimX>(ddc::DiscreteElement<DDimX>(0), ddc::DiscreteVector<DDimX>(3)),ddc::DiscreteDomain<DDimY>(ddc::DiscreteElement<DDimY>(0), ddc::DiscreteVector<DDimY>(3)));
  ddc::Chunk src("src", dom, ddc::DeviceAllocator<double>());
  ddc::for_each(dom, [&](ddc::DiscreteElement<DDimX, DDimY> ixy) {
            ddc::DiscreteElement<DDimX> ix(ixy);
            ddc::DiscreteElement<DDimY> iy(ixy);
            src(ix, iy) = ix.uid() + 0.1 * iy.uid();
        });
    ddc::Chunk dst("dst", dom, ddc::DeviceAllocator<double>());
    ddc::parallel_deepcopy(dst, src);
    print_2d("src                           ", src);
    print_2d("dst after deepcopy", dst);
    ddc::ChunkSpan dst_span = dst.span_view();
    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            dst.domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> ixy) {
                ddc::DiscreteElement<DDimX> ix(ixy);
                ddc::DiscreteElement<DDimY> iy(ixy);
                dst_span(ix, iy) *= 2.0;
            });
    print_2d("dst after parallel *= 2", dst);
  }
  {
    ddc::DiscreteDomain<DDimX,DDimY> doma(ddc::DiscreteElement<DDimX,DDimY>(10,20),ddc::DiscreteVector<DDimX,DDimY>(3,3));
    ddc::Chunk dc("Name",doma,ddc::DeviceAllocator<double>());
    ddc::ChunkSpan span = dc.span_view();
    ddc::for_each(span.domain(),[&](ddc::DiscreteElement<DDimX,DDimY> ixy){
      ddc::DiscreteElement<DDimX> ix(ixy);
      ddc::DiscreteElement<DDimY> iy(ixy);
      span(ix,iy) = ix.uid() + iy.uid() ;
    });
    std::cout<<span(ddc::DiscreteElement<DDimX>(11),ddc::DiscreteElement<DDimY>(21))<<std::endl;
    ddc::DiscreteDomain<DDimX> sub(ddc::DiscreteElement<DDimX>(11),ddc::DiscreteVector<DDimX>(2));
    auto sub_dom = span[sub];
    std::cout<<"   sliced at X = 11, to X = 12 size----->  "<<sub_dom.extent<DDimX>()<<std::endl;
  }
  {
    auto const x_domain = ddc::init_discrete_space<DDimX>(DDimX::init<DDimX>(ddc::Coordinate<X>(0.0),ddc::Coordinate<X>(1.0),ddc::DiscreteVector<DDimX>(50)));
    auto const y_domain = ddc::init_discrete_space<DDimY>(DDimY::init<DDimY>(ddc::Coordinate<Y>(0.0),ddc::Coordinate<Y>(1.0),ddc::DiscreteVector<DDimY>(50)));
    std::cout<<" X 5 points x:[0.0,1.0]  -------------->   "<<ddc::step<DDimX>()<<"\n";
    std::cout<<" Y 5 points y:[0.0,1.0] --------------->   "<<ddc::step<DDimY>()<<"\n";
    
    ddc::for_each(x_domain,[&](ddc::DiscreteElement<DDimX> ix){
      ddc::Coordinate<X> x = ddc::coordinate(ix);
      std::cout<<"  ix =    "<<ddc::select<DDimX>(ix)<<"  -> "<<ddc::get<X>(x)<<"\n";
    });
    ddc::DiscreteDomain<DDimX,DDimY> dom_xy(x_domain,y_domain);
    ddc::Chunk field("FIELD", dom_xy,ddc::DeviceAllocator<double>());
    ddc::ChunkSpan span = field.span_view();
    ddc::for_each(dom_xy,[&](ddc::DiscreteElement<DDimX,DDimY> ixy){
      double x = ddc::get<X>(ddc::coordinate(ddc::DiscreteElement<DDimX>(ixy)));
      double y = ddc::get<Y>(ddc::coordinate(ddc::DiscreteElement<DDimY>(ixy)));
      span(ixy) = x*x + y*y;
    });
    print_2d("f(x) = x^2 + y^2",span);
  }
  std::cout<<"\n";
  
  
}
