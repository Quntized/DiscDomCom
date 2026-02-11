#include <ddc/ddc.hpp>
#include <iostream>

struct TagX {};
struct TagY {};
int main() {
  ddc::DiscreteDomain<TagX,TagY> dom(ddc::DiscreteElement<TagX,TagY>(1,2),ddc::DiscreteVector<TagX,TagY>(10,20));
  ddc::Chunk<double , ddc::DiscreteDomain<TagX,TagY> >cd("Data",dom);
  ddc::for_each(dom,[&](ddc::DiscreteElement<TagX,TagY> elem) {
    //cd(elem) = ddc::get<TagX>(elem) + ddc::get<TagY>(elem);  // Doesn't work the ddc::get<Tag>(elem) on discreteelement 
    auto idx_x = ddc::select<TagX>(elem).uid();
    auto idx_y = ddc::select<TagY>(elem).uid();
    cd(elem) = idx_x + idx_y;
  });
  std::cout << "DOmain_size: "<< dom.size() << std::endl;
  std::cout<< "First Element: "<< cd(dom.front()) << std::endl;
  ddc::for_each(dom , [&](ddc::DiscreteElement<TagX,TagY> elem){
    std::cout<<"element: "<<elem<<" -- "<<cd(elem)<<std::endl;
  });



  return 0;
}
