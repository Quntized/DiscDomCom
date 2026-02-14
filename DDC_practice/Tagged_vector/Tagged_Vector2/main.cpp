#include <iostream> 
#include <type_traits>
#include <ddc/ddc.hpp>
struct X;
struct Y;
struct Z;
struct W;

int main(int argc, char** argv){
  Kokkos::ScopeGuard kokkos_scope(argc,argv);
  {
  using Vec2D = ddc::detail::TaggedVector<double,X,Y>;
  std::cout<<std::boolalpha;
  std::cout<< "Is int taggedVector? "<<ddc::detail::is_tagged_vector_v<int><<"\n";
  std::cout<<"Is Vec2D a TaggedVector? "<<ddc::detail::is_tagged_vector_v<Vec2D><<"\n";
  std::cout<< "Is double taggedVector? "<<ddc::detail::is_tagged_vector_v<double><<"\n";

  //
  using Vec3D = ddc::detail::TaggedVector<double,X,Y,Z>;
  //using ExtractedSeq = ddc::detail::to_type_seq_t<Vec3D>;
  //using ExpectedSeq = ddc::detail::TypeSeq<X,Y,Z>;
  //std::cout<<std::boolalpha;
  //std::cout<<"Extracted Seq matches Expected:      "<<std::s_same_v<ExtractedSeq,ExpectedSeq> <<"\n";
  ddc::detail::TaggedVector<double,X,Y,Z> vec(1.0,2.0,3.0);
  std::cout<<"vec x component = "<<ddc::get<X>(vec)<<"\n";
  std::cout<<"vec x component using vec.get<X>() = "<<vec.get<X>()<<"\n";
  std::cout<<"vec y component = "<<ddc::get<Y>(vec)<<"\n";
  std::cout<<"vec z component = "<<ddc::get<Z>(vec)<<"\n";
  ddc::get<X>(vec) = 12.0;
  std::cout<<"vec x component after modifying = "<<ddc::get<X>(vec)<<"\n";
  // i can use get_or if tags non_existing means --->
  std::cout<<"This has X tag exists ---> "<<ddc::get_or<X>(vec,14.4)<<"\n";
  std::cout<<"This isn't any tag , so created new W= "<<ddc::get_or<W>(vec,88.8)<<"\n";
  ddc::detail::TaggedVector<double,X,Y,Z> vec1(-6.0,2.0,3.0);
  std::cout<<"vec1 x component = "<<ddc::get<X>(vec1)<<"\n";
  std::cout<<"vec1 y component = "<<ddc::get<Y>(vec1)<<"\n";
  std::cout<<"vec1 z component = "<<ddc::get<Z>(vec1)<<"\n";
  auto pos = + vec1;
  std::cout<<"vec1 x component after pos = +vec = "<<ddc::get<X>(pos)<<"\n";
  std::cout<<"vec1 y component after pos = +vec = "<<ddc::get<Y>(pos)<<"\n";
  std::cout<<"vec1 z component after pos = +vec = "<<ddc::get<Z>(pos)<<"\n";
  auto neg = - vec1;
  std::cout<<"vec1 x component after neg = -vec = "<<ddc::get<X>(neg)<<"\n";
  std::cout<<"vec1 y component after neg = -vec = "<<ddc::get<Y>(neg)<<"\n";
  std::cout<<"vec1 z component after neg = -vec = "<<ddc::get<Z>(neg)<<"\n";
  ddc::detail::TaggedVector<double,X,Y,Z> vec2(1.0,2.0,3.0);
  ddc::detail::TaggedVector<double,X,Y,Z> vec3(3.0,5.0,8.0);
  auto sum = vec2 + vec3;
  std::cout<<"sum x component = "<<ddc::get<X>(sum)<<"\n";
  std::cout<<"sum y component = "<<ddc::get<Y>(sum)<<"\n";
  std::cout<<"sum z component = "<<ddc::get<Z>(sum)<<"\n";
  ddc::detail::TaggedVector<double,X,Y,Z> vec4(1.0,2.0,3.0);
  ddc::detail::TaggedVector<int,X,Y,Z> vec5(3.0,5.0,8.0);
  auto sum2 = vec4+ vec5;
  std::cout<<"sum x component = "<<ddc::get<X>(sum2)<<"\n";
  std::cout<<"sum y component = "<<ddc::get<Y>(sum2)<<"\n";
  std::cout<<"sum z component = "<<ddc::get<Z>(sum2)<<"\n";
  ddc::detail::TaggedVector<double,X> v(1.0);
  auto result = v + 3.0;
  ddc::detail::TaggedVector<double , X> result2 = 3.0 + v;
  std::cout<<"result of v+3, x component = "<<ddc::get<X>(result)<<"\n";
  std::cout<<"result of 3+v, x component = "<<ddc::get<X>(result2)<<"\n";
  double scalar = 5.0;
  ddc::detail::TaggedVector<double,X,Y,Z> vd(4.0,5.0,6.0);
  ddc::detail::TaggedVector<double,X,Y,Z> reslt(ddc::get<X>(vd) + scalar,ddc::get<Y>(vd) + scalar,ddc::get<Z>(vd) + scalar);
  std::cout << "  X: " << ddc::get<X>(reslt) << "\n";
  std::cout << "  Y: " << ddc::get<Y>(reslt) << "\n";
  std::cout << "  Z: " << ddc::get<Z>(reslt) << "\n";
  ddc::detail::TaggedVector<double,X,Y,Z> re = vd;
  re += 3.0;
  std::cout << "  X: " << ddc::get<X>(re) << "\n";
  std::cout << "  Y: " << ddc::get<Y>(re) << "\n";
  std::cout << "  Z: " << ddc::get<Z>(re) << "\n";
  ddc::detail::TaggedVector<double,X,Y,Z> v2(12.0,13.0,21.0);
  ddc::detail::TaggedVector<double,X,Y,Z> v3(18.0,11.0,51.0);
  auto diff = v3 - v2;
  std::cout << "  X: " << ddc::get<X>(diff) << "\n";
  std::cout << "  Y: " << ddc::get<Y>(diff) << "\n";
  std::cout << "  Z: " << ddc::get<Z>(diff) << "\n";
  auto r = v3;
  r -= 100;
  std::cout << "  X: " << ddc::get<X>(r) << "\n";
  std::cout << "  Y: " << ddc::get<Y>(r) << "\n";
  std::cout << "  Z: " << ddc::get<Z>(r) << "\n";
  ddc::detail::TaggedVector<double,X,Y,Z> v4(18.0,11.0,51.0);
  v4 =  2.0*v4;
  //sc *=4.0;
  std::cout << "  X: " << ddc::get<X>(v4) << "\n";
  std::cout << "  Y: " << ddc::get<Y>(v4) << "\n";
  std::cout << "  Z: " << ddc::get<Z>(v4) << "\n";
  ddc::detail::TaggedVector<double,X,Y,Z,W> v6(1.0,2.0,3.0,4.0);
  auto vec_xz = ddc::select<X,Z>(v6);
  std::cout<<"vecxz(x_component)  =  "<<ddc::get<X>(vec_xz)<<std::endl;
  std::cout<<"vecxz(y_component)  =  "<<ddc::get<Z>(vec_xz)<<std::endl;
  auto vec_yz = ddc::select<Y,Z>(v6);
  std::cout<<"vecyz(z_component)  =  "<<ddc::get<Z>(vec_yz)<<std::endl;
  std::cout<<"vecyz(y_component)  =  "<<ddc::get<Y>(vec_yz)<<std::endl;
  auto found_x = ddc::detail::take<X>(v6);
  std::cout<<"found_x = "<<ddc::get<X>(found_x)<<"\n";
  ddc::detail::TaggedVector<double, X> vec_x(1.0);
  ddc::detail::TaggedVector<double, Y> vec_y(12.0);
  ddc::detail::TaggedVector<double, Z> vec_z(11.0);
  auto foun_x = ddc::detail::take<X>(vec_x,vec_y,vec_z);
  std::cout<<"found_x = "<<ddc::get<X>(foun_x)<<"\n";
  std::cout<<"v6.size -- >"<<v6.size()<<"\n";
  std::cout<<"vec4.size -- >"<<decltype(vec4)::size()<<"\n";
  ddc::detail::TaggedVector<double,X> v7;
  std::cout<<"v6 X value = "<<ddc::get<X>(v7)<<"\n";
  ddc::detail::TaggedVector<double,X,Y,Z> v8 = vec4;
  std::cout<<"v8 X value = "<<ddc::get<X>(v8)<<"\n";
  std::cout<<"v8 Y value = "<<ddc::get<Y>(v8)<<"\n";
  std::cout<<"v8 Z value = "<<ddc::get<Z>(v8)<<"\n";
  ddc::detail::TaggedVector<double,X,Y,Z,W> v9 = std::move(v6);
  std::cout<<"v9 X value = "<<ddc::get<X>(v9)<<"\n";
  std::cout<<"v9 Y value = "<<ddc::get<Y>(v9)<<"\n";
  std::cout<<"v9 Z value = "<<ddc::get<Z>(v9)<<"\n";
  }
  ddc::detail::TaggedVector<double, X> vec_x(1.0);
  ddc::detail::TaggedVector<double, Y> vec_y(2.0);
  ddc::detail::TaggedVector<double, Z> vec_z(3.0);
  ddc::detail::TaggedVector<double, X, Y, Z> vec_3d(vec_x, vec_y, vec_z);
  std::cout << "Combined from 1D vectors:\n";
  std::cout << "  (" << ddc::get<X>(vec_3d) << ", "
              << ddc::get<Y>(vec_3d) << ", "
              << ddc::get<Z>(vec_3d) << ")\n\n";
  ddc::detail::TaggedVector<double, X, Y> vec_xy(10.0, 20.0);
  ddc::detail::TaggedVector<double, Z> vec_z2(30.0);
  ddc::detail::TaggedVector<double, X, Y, Z> vec_3d_alt(vec_xy, vec_z2);
  std::cout << "  (" << ddc::get<X>(vec_3d_alt) << ", "
              << ddc::get<Y>(vec_3d_alt) << ", "
              << ddc::get<Z>(vec_3d_alt) << ")\n\n";
  ddc::detail::TaggedVector<double, X, Y, Z> vec_reorder(vec_z, vec_x, vec_y);
  ddc::detail::TaggedVector<double,X,Y,Z> vec(23.0,43.0,54.0);
  auto& arr = vec.array();
  for(std::size_t i = 0; i<arr.size(); ++i){
    std::cout<<"        arr["<<i<<"]"<<" = "<<arr[i]<<"\n";
  }
    arr[0] = 2.0;
    arr[1] = 1.0;
    arr[2] = 0.0;
    std::cout<<" arr[0] using ddc::get = "<<ddc::get<X>(vec)<<"\n";
    std::cout<<" arr[1] using ddc::get = "<<ddc::get<Y>(vec)<<"\n";
    std::cout<<" arr[2] using ddc::get = "<<ddc::get<Z>(vec)<<"\n";
    ddc::detail::TaggedVector<double,X,Y,Z> vec1(23.0,45.0,54.0);
    std::cout<<" vec1[0] = "<<vec1[0]<<std::endl;
    std::cout<<" vec1[1] = "<<vec1[1]<<std::endl;
    std::cout<<" vec1[0] = "<<vec1[2]<<std::endl;
    auto v = ddc::get<X>(vec1);
    std::cout<<(vec[0] == v ?  "TRUE": "FALSE")<<"\n";
    std::cout<<(vec1 == vec? "True": "False")<<std::endl;
    std::cout<<(vec1 != vec? "True": "False")<<std::endl;
    vec1.get<X>() = 299.0;
    std::cout<<"ddc::get<X>(vec1) = "<<ddc::get<X>(vec1)<<std::endl;
    ddc::detail::TaggedVector<double , X , Y ,Z> vv(23.0,54.0,78.0);
    std::cout<<" X :  using get_or() = "<<ddc::get_or<X>(vv,-999.999)<<std::endl;
    std::cout<<" Y :  using get_or() = "<<ddc::get_or<Y>(vv,-999.999)<<std::endl;
    std::cout<<" Z :  using get_or() = "<<ddc::get_or<Z>(vv,-999.999)<<std::endl;
    std::cout<<" W :  using get_or() = "<<ddc::get_or<W>(vv,-999.999)<<std::endl;
    std::cout<<" X :  using vv.get_or() = "<<vv.get_or<X>(-999.999)<<std::endl;
    ddc::detail::TaggedVector<double,X> vec_val(12.0);
    double val = vec_val.value();
   // ddc::detail::TaggedVector<double,X> vec_val(12.0);
   std::cout<<val<<"\n";
  return 0;
}


