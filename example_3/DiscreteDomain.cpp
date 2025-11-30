#include <array>
#include <iostream>
#include <type_traits>
#include <vector>

#define KOKKOS_FUNCTION
#define KOKKOS_DEFAULTED_FUNCTION

template <class T>
inline constexpr bool is_discrete_domain_v = false;
class Domain1D{
  int m_front;
  int m_extent;
public:
  Domain1D(int f =0, int e = 0) : m_front(f), m_extent(e) {}
  int front() const {return m_front;}
  int extents() const {return m_extent;}
  friend int operator+(Domain1D const& d, int rhs) {return d.front() + rhs;}
};
template <> inline constexpr bool is_discrete_domain_v<Domain1D> = true;
template <class... DDims>
struct DiscreteElement{
  std::vector<int> vals;
  template <class... Vs>
  DiscreteElement(Vs... vs) : vals{static_cast<int>(vs)...} {}
  DiscreteElement operator+(DiscreteElement const& other) const {
    DiscreteElement r;
    r.vals.resize(std::max(vals.size(),other.vals.size()));
    for (size_t i = 0; i<r.vals.size(); ++i){
      int a = i < vals.size() ? vals[i] : 0;
      int b = i < other.vals.size() ? other.vals[i] : 0;
      r.vals[i] = a + b;
    }
    return r;
  }
  DiscreteElement() = default;
};
template <class... Ts>
std::ostream& operator<<(std::ostream& os , DiscreteElement<Ts...> const& e){
  os << "(";
  for (size_t i = 0; i<e.vals.size(); ++i){
    if (i) os << ",";
    os<<e.vals[i];
  }
  os<<")";
  return os;
}
template <class... DDims>
class DiscreteDomain
{
public:
  using discrete_element_type = DiscreteElement<DDims...>;
  discrete_element_type m_element_begin;
  discrete_element_type m_element_end;
  template <class... DDoms , class=std::enable_if_t<(is_discrete_domain_v<DDoms> && ...)>>
  KOKKOS_FUNCTION constexpr explicit DiscreteDomain(DDoms const&... domains): m_element_begin(domains.front()...),m_element_end(domains.front()+domains.extents()...)
  {}
};
int main(){
  Domain1D dx(0,10);
  Domain1D dy(5,3);
  struct X{}; struct Y{};
  DiscreteDomain<X,Y> combined(dx,dy);
  std::cout<<"Combined.m_element_begin = "<<combined.m_element_begin<<"\n";
  std::cout<<"Combined.m_element_end = "<<combined.m_element_end<<"\n";
  
  Domain1D dz(2,4);
  struct Z {};
  DiscreteDomain<X,Y,Z> combined3(dx,dy,dz);
  std::cout<<"combined3.begin = "<<combined3.m_element_begin<<"\n";
  std::cout<<"combined3.end = "<<combined3.m_element_end<<"\n";


}
