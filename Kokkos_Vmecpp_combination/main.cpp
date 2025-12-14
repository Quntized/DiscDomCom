#include "plasma_lib.hpp"

int main() {
    PlasmaGenerator generator;
    PlasmaConfig my_shape;
    my_shape.major_radius = 3.0;
    my_shape.elongation   = 1.7;
    my_shape.triangularity= 0.4;
    my_shape.nfp          = 1;  
    generator.set_config(my_shape);
    generator.generate_vmec_file("vmec.test_shape");
    return 0;
}
