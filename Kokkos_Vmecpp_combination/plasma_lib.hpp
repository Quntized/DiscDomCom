#ifndef PLASMA_LIB_HPP
#define PLASMA_LIB_HPP
#include <string>
#include <vector>
struct PlasmaConfig {
    double major_radius = 5.5;
    double minor_radius = 0.5;
    double elongation = 1.0; 
    double triangularity = 0.0;
    int nfp = 1;          
    double flux = -1.74;    
    int grid_points = 64;     
    int fourier_modes = 12; 
};
class PlasmaGenerator {
public:
    PlasmaGenerator();
    void set_config(const PlasmaConfig& config);
    void generate_vmec_file(const std::string& filename);
private:
    PlasmaConfig cfg;
    void write_to_disk(const std::string& fname, 
                       const std::vector<double>& rbc, 
                       const std::vector<double>& zbs);
};
#endif
