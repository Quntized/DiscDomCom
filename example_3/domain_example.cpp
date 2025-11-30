#include <cassert>
#include <iostream>
#include <type_traits>
#include <vector>
#include <initializer_list>

// ------------------ simple tags ------------------
struct X {};
struct Y {};
struct Z {};

// ------------------ compile-time utilities ------------------
template <typename Dim, typename... Pack>
struct index_of_impl;

template <typename Dim>
struct index_of_impl<Dim> { static constexpr int value = -1; };

template <typename Dim, typename Head, typename... Tail>
struct index_of_impl<Dim, Head, Tail...> {
    static constexpr int next = index_of_impl<Dim, Tail...>::value;
    static constexpr int value = std::is_same<Dim, Head>::value ? 0
        : (next == -1 ? -1 : 1 + next);
};

template <typename Dim, typename... Pack>
constexpr int index_of_v = index_of_impl<Dim, Pack...>::value;

template <typename Dim, typename... Pack>
constexpr bool has_dim_v = (index_of_v<Dim, Pack...> != -1);

// ------------------ DiscreteElement ------------------
template <typename... Tags>
struct DiscreteElement {
    std::vector<int> vals;

    DiscreteElement() = default;

    template <typename... Vs, typename = std::enable_if_t<sizeof...(Vs) == sizeof...(Tags)>>
    DiscreteElement(Vs... vs) : vals{static_cast<int>(vs)...} {}

    // construct single-dim DiscreteElement<Dim> from multi-dim source
    template <typename... SourceTags, typename = std::enable_if_t<sizeof...(Tags) == 1>>
    DiscreteElement(DiscreteElement<SourceTags...> const& src) {
        using Dim = typename std::tuple_element<0, std::tuple<Tags...>>::type;
        constexpr int idx = index_of_v<Dim, SourceTags...>;
        static_assert(idx != -1, "Source doesn't contain requested dimension");
        vals.resize(1);
        vals[0] = src.vals[idx];
    }

    template <typename Dim>
    int get() const {
        constexpr int idx = index_of_v<Dim, Tags...>;
        static_assert(idx != -1, "Requested dimension not present in this DiscreteElement");
        return vals[idx];
    }

    DiscreteElement operator+(DiscreteElement const& other) const {
        DiscreteElement r;
        r.vals.resize(std::max(vals.size(), other.vals.size()));
        for (size_t i = 0; i < r.vals.size(); ++i) {
            int a = i < vals.size() ? vals[i] : 0;
            int b = i < other.vals.size() ? other.vals[i] : 0;
            r.vals[i] = a + b;
        }
        return r;
    }

    bool operator<=(DiscreteElement const& o) const {
        if (vals.size() != o.vals.size()) return false;
        for (size_t i = 0; i < vals.size(); ++i)
            if (vals[i] > o.vals[i]) return false;
        return true;
    }
    bool operator>=(DiscreteElement const& o) const {
        if (vals.size() != o.vals.size()) return false;
        for (size_t i = 0; i < vals.size(); ++i)
            if (vals[i] < o.vals[i]) return false;
        return true;
    }
};

template <typename... Tags>
std::ostream& operator<<(std::ostream& os, DiscreteElement<Tags...> const& e) {
    os << "(";
    for (size_t i = 0; i < e.vals.size(); ++i) {
        if (i) os << ", ";
        os << e.vals[i];
    }
    os << ")";
    return os;
}

// ------------------ DiscreteDomain ------------------
template <typename... Tags>
struct DiscreteDomain {
    using element_t = DiscreteElement<Tags...>;
    element_t begin_v;
    element_t end_v;

    DiscreteDomain() = default;
    DiscreteDomain(element_t b, element_t e) : begin_v(b), end_v(e) {}

    element_t extents() const {
        element_t r;
        r.vals.resize(begin_v.vals.size());
        for (size_t i = 0; i < r.vals.size(); ++i) r.vals[i] = end_v.vals[i] - begin_v.vals[i];
        return r;
    }

    // The corrected restrict_with implementation
    template <typename... ODTags>
    auto restrict_with(DiscreteDomain<ODTags...> const& odomain) const {
        // Assert odomain is inside this domain for shared dims
        bool ok = true;
        (void)std::initializer_list<int>{
            ( (index_of_v<ODTags, Tags...> != -1)
                ? ( ( this->begin_v.template get<ODTags>() <= odomain.begin_v.template get<ODTags>() ) ? 0 : (ok = false, 0) )
                : 0 )...
        };
        (void)std::initializer_list<int>{
            ( (index_of_v<ODTags, Tags...> != -1)
                ? ( ( this->end_v.template get<ODTags>() >= odomain.end_v.template get<ODTags>() ) ? 0 : (ok = false, 0) )
                : 0 )...
        };
        assert(ok && "odomain must be inside this domain for shared dimensions");

        element_t myext = this->extents();
        element_t ret_begin; ret_begin.vals.resize(sizeof...(Tags));
        element_t ret_ext;   ret_ext.vals.resize(sizeof...(Tags));

        // fill per Tag in Tags...
        size_t idx = 0;
        auto set_component = [&](auto tag) {
            using Tag = decltype(tag);
            if constexpr (index_of_v<Tag, ODTags...> != -1) {
                ret_begin.vals[idx] = odomain.begin_v.template get<Tag>();
                ret_ext.vals[idx]   = odomain.extents().template get<Tag>();
            } else {
                ret_begin.vals[idx] = this->begin_v.template get<Tag>();
                ret_ext.vals[idx]   = myext.template get<Tag>();
            }
            ++idx;
        };
        (void)std::initializer_list<int>{ ( set_component(Tags{}), 0 )... };

        element_t ret_end = ret_begin + ret_ext;
        return DiscreteDomain<Tags...>(ret_begin, ret_end);
    }
};

// ------------------ Demo main ------------------
int main() {
    DiscreteElement<X,Y,Z> my_begin(0,10,20);
    DiscreteElement<X,Y,Z> my_end(10,15,23);
    DiscreteDomain<X,Y,Z> myDomain(my_begin, my_end);

    DiscreteElement<Y,Z> od_begin(11,21);
    DiscreteElement<Y,Z> od_end(14,23);
    DiscreteDomain<Y,Z> odomain(od_begin, od_end);

    std::cout << "this domain begin = " << myDomain.begin_v << " end = " << myDomain.end_v << "\n";
    std::cout << "other domain begin = " << odomain.begin_v << " end = " << odomain.end_v << "\n";

    auto restricted = myDomain.restrict_with(odomain);

    std::cout << "restricted begin = " << restricted.begin_v << " end = " << restricted.end_v << "\n";
    return 0;
}

