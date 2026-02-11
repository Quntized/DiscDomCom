#include <ddc/ddc.hpp>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstring>

struct X {};
struct Y {};

int main() {
ddc::DiscreteDomain<X,Y> dom(ddc::DiscreteElement<X,Y>(0,0),ddc::DiscreteVector<X,Y>(5,4));
ddc::Chunk<double , ddc::DiscreteDomain<X,Y>> temperature("Temperature",dom);
ddc::Chunk<double , ddc::DiscreteDomain<X,Y>> pressure("pressure",dom);
ddc::for_each(dom , [&](auto elem){
  auto i = ddc::select<X>(elem).uid();
  auto j = ddc::select<Y>(elem).uid();
  temperature(elem) = i + 1*j ; 
  pressure(elem) = i+j+1;
  std::cout<<"i "<< i << " j "<< j<<" = "<<temperature(elem)<<std::endl; 
});
std::cout<< "Chunk 1 label "<< temperature.label() << std::endl;
std::cout<<"Chunk 2 label"<< pressure.label()<<std::endl;
double* temp_ptr = temperature.data_handle();
const double* pressure_ptr = pressure.data_handle();
std::cout<<"Memory Address of temperature pointer = "<<temp_ptr << std::endl;
std::cout<< "Memory Address of pressure pointer = " << pressure_ptr << std::endl;
for (int i = 0; i<dom.size(); i++){
  std::cout<< "temp_ptr[ "<< i << " ] = " << temp_ptr[i] << std::endl;
}
ddc::for_each(temperature.domain() , [&](auto elem){
  auto i = ddc::select<X>(elem).uid();
  auto j = ddc::select<Y>(elem).uid();
  std::cout<<"i "<< i << " j "<< j<<" = "<<temperature(elem)<<std::endl; 
});
std::cout<<dom.size()<<std::endl;
//double* temp_ptr = temperature.data_handle();
for(std::size_t i = 0; i< dom.size(); ++i){
  temp_ptr[i] += 10.0;
}
ddc::for_each(temperature.domain() , [&](auto elem){
  auto i = ddc::select<X>(elem).uid();
  auto j = ddc::select<Y>(elem).uid();
  std::cout<<"i "<< i << " j "<< j<<" = "<<temperature(elem)<<std::endl; 
});

for (int i = 0; i<dom.size(); i++){
  std::cout<< "temp_ptr[ "<< i << " ] = " << temp_ptr[i] << std::endl;
}
std::cout<< "First Element = "<< temperature(dom.front())<< std::endl;
const double* pressure_data = pressure.data_handle();
auto [min_it,max_it] = std::minmax_element(pressure_data,pressure_data + dom.size());
std::cout<< "Min " <<*min_it<<std::endl;
std::cout<<"Max "<< *max_it<<std::endl;
double sum = std::accumulate(pressure_data, pressure_data+dom.size(),0.0);
double mean = sum / dom.size();
std::cout<<"Mean " <<mean<<std::endl;
auto temp_mdspan = temperature.allocation_mdspan();
auto press_mdspan = pressure.allocation_mdspan();
std::cout<< "MdSpan rank " << temp_mdspan.rank()<<std::endl;
std::cout<<"MdSpan extents(0) "<<temp_mdspan.extent(0)<<std::endl;
std::cout<<"MdSpan extent(1) "<<temp_mdspan.extent(1)<<std::endl;
std::cout<<"temp at mdspan[2][1] = " <<temp_mdspan(2,1)<<std::endl;
temp_mdspan(2,1)= 999.9;
std::cout<<"After modification"<<temp_mdspan(2,1)<<std::endl;
ddc::Chunk<double , ddc::DiscreteDomain<X,Y>> backup("Backum",dom);
std::memcpy(backup.data_handle(),temperature.data_handle(),dom.size() *sizeof(double));
std::cout<<"Backup[0][0] "<<backup(dom.front())<<std::endl;
std::cout<<"Back up label "<< backup.label() <<std::endl;
ddc::Chunk<double, ddc::DiscreteDomain<X,Y>> zeros("Zeros", dom);
std::fill(zeros.data_handle(),zeros.data_handle()+dom.size(),0.0);
std::cout<< "First element : "<< zeros(dom.front())<<std::endl;
std::cout<<"Last element: "<< zeros(dom.back())<<std::endl;
auto compute_sum = [](const double* data , std::size_t n ) -> double {
  double total = 0.0;
  for (std::size_t i = 0; i<n; ++i){
    total += data[i];
  }
  return total;
};
double temp_sum = compute_sum(temperature.data_handle(), dom.size());
std::cout<<"sum of all temperature = "<< temp_sum <<std::endl;
bool are_equal = std::equal(temperature.data_handle(), temperature.data_handle() + dom.size(),backup.data_handle());
std::cout<<"Are equal "<<are_equal<<std::endl;
double* data = temperature.data_handle();
auto start_idx = 0;
auto end_idx= dom.size();
for(std::size_t i = start_idx; i<end_idx; ++i){
  data[i] = data[i]*1.1;
}
std::cout<<"FIrst elelemt after applying 1.1 * data[i] "<<temperature(dom.front())<<std::endl;


  return 0;
}
