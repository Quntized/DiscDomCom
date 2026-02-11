#ifndef PLASMA_LIB_HPP
#define PLASMA_LIB_HPP

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>

struct PlasmaConfig {
    // --- 1. Grid & Symmetry ---
    bool lasym = false;
    int nfp = 5;            // 5 Field Periods (W7-X)
    int nzeta = 36;
    int mpol = 8;
    int ntor = 1;

    // --- 2. Iteration Control (ROBUST MULTI-GRID) ---
    // Vectors support coarse-to-fine grid steps
    std::vector<int> ns_array = {15, 31};      
    std::vector<double> ftol_array = {1e-6, 1e-8};
    std::vector<int> niter_array = {2000, 4000};
    
    int nstep = 200;
    double delt = 0.7;
    bool lfreeb = false;

    // --- 3. Physics Control ---
    double phiedge = -1.0; 
    int ncurr = 1;
    double gamma = 0.0;

    // --- 4. Mass/Pressure Profile ---
    // 'two_power' requires exactly 3 coefficients: {Amp, a, b}
    std::vector<double> am = {1.0, 2.0, 1.0}; 
    double pres_scale = 1.0; 
    double spres_ped = 1.0;

    // --- 5. Current Profile ---
    // 'two_power' requires exactly 3 coefficients: {Amp, a, b}
    std::vector<double> ac = {1.0, 2.0, 1.0}; 
    double curtor = 0.0;
    double bloat = 1.0;

    // --- 6. Iota Profile ---
    // Uses 'power_series' (flexible size)
    std::vector<double> ai = {0.0}; 

    // --- 7. Geometry Generation ---
    double R0 = 5.5;
    double a = 0.5;
    double kappa = 1.0;
    double delta = 0.0;
    double torsion = 0.0;
    double axis_shift_amp = 0.0;
    
    // Axis Guess
    std::vector<double> raxis = {5.5};
    std::vector<double> zaxis = {0.0};

    // FFT Settings
    int resolution = 128;
    double detect_tol = 1e-9;
};
class PlasmaLib {
public:
    PlasmaLib();
    void set_config(const PlasmaConfig& config);
    void generate_file(const std::string& filename);

private:
    PlasmaConfig cfg;
    void write_vmec_format(const std::string& fname, 
                           int max_m, int max_n, 
                           const std::map<std::pair<int,int>, double>& rbc,
                           const std::map<std::pair<int,int>, double>& zbs);
                           
    // UPDATED HELPER: Uses commas for VMEC++ compliance
    template <typename T>
    void write_array(std::ostream& os, const std::string& name, const std::vector<T>& arr) {
        os << name << " = ";
        for (size_t i = 0; i < arr.size(); ++i) {
            os << arr[i] << (i < arr.size() - 1 ? ", " : ",\n");
        }
    }
};
#endif
