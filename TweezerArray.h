#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // transform
#include <iterator> 
#include <type_traits>
#include <execution> // par_unseq

#include <Windows.h>

#include "Parameters.h"
#include "Tweezer.h"
#include "basic_fileIO.h" // Just for debugging, can be removed later
#include "math_utils.cuh"

#include "statusBox.h"






class TweezerArray {
	public:
		explicit TweezerArray(
            const Parameters& params, statusBox* box
        );


        // This could be noexcept because the mask creation already checks if the correct
        // number of peaks is detected
        void update_position_in_camera_image(
            const std::vector<unsigned int>& sorted_flattened_peak_indices
        ) noexcept;

        // Extract the current amplitude/intensity and phase from the fft array
        // If the fix_phase flag is set the phase will not be updated
        void update_current_intensities_and_phases(
            const cufftDoubleComplex* fft_output,
            bool fix_phase
        ) noexcept;

        // Extract the current intensity from the camera image
        // This function is largely equal to 
        void update_current_intensities_from_camera_image(
            const byte* camera_image
        );

        void update_plot_positions(int center, int spacing_x_px, int offset_x,
                              int spacing_y_px, int offset_y);

        // In CGHAlgorithm the main WGS loop calculated the weights which are used
        // to update the target amplitude of each tweezer
        void update_target_intensities(
            const std::vector<double>& weights
        );

        // Write amplitudes and phases to the respective sites in the fft array
        void update_fft_array(
            cufftDoubleComplex* fft_array
        ) const noexcept;

        // Mean array intensity. The local intensities are calculated by a local gaussian average
        // on the respective array sites.
        double get_mean_intensity(
            void
        ) const noexcept;


        double get_mean_amplitude(
            void
        ) const noexcept;

        // Standard deviation of the tweezer array, normalized to the mean
        double get_nonuniformity(
            void
        ) const noexcept;

        // Vector of local intensities
        // I guess returning the vector would als be fine as we will probably
        // never get to millions of tweezers
        std::vector<double> get_intensities(
            void
        ) const noexcept;


        std::vector<double> get_amplitudes(
            void
        ) const noexcept;

        // Number of tweezers
        unsigned int get_array_size(
            void
        ) const noexcept;
        int get_x_plot_start() { return x_start_plot; }
        int get_x_plot_stop() { return x_stop_plot; }
        int get_y_plot_start() { return y_start_plot; }
        int get_y_plot_stop() { return y_stop_plot; }

    private:
        // 
    statusBox *editT;
    const unsigned int number_of_pixels_padded;
    const unsigned int number_of_pixels_unpadded;

    unsigned int num_traps_x, num_traps_y, num_traps;
    double spacing_x_um, spacing_y_um;

    // Optical parameters and derived quantities
    double focal_length_um, wavelength_um;

    // Resolutions in padded Fourier space when delta k
    // is transformed by the scale factor wavelenghth * focal_length
    double delta_x_padded_um;

    int y_start_plot, y_stop_plot, x_start_plot, x_stop_plot;

    // waist_px also declared as double to avoid rounding errors if possible
    double waist_um, waist_px_in_fft, waist_px_in_camera_image;

    // Mean intensity of all local intensities in the tweezer array
    double mean_intensity, mean_amplitude, standard_deviation;

    std::string geometry;

    // Main data structure to operate on
    std::vector<Tweezer> tweezer_vec;

    bool position_in_camera_image_set;

    // This is a template because I used it to calculate the local intensity in both the fft_array and the 
    // camera image. I removed the local intensity calculation for the theoretical iteration because it's
    // not necessary and slows things down dramatically. So this function is only used for unsigned chars 
    // at this point
    template<typename T>
    double get_local_intensity(
        const T* arr,
        int x, int y, int width, int height,
        int radius,
        const std::vector<double>& gaussian_weights
    ) const {
        // Indices are defined as ints so negative values can wrap around 
        // when mod number_of_pixels_padded

        if (gaussian_weights.size() != std::pow(2 * radius + 1, 2)) {
            throw std::length_error("TweezerArray::get_local_intensity: gaussian_weights invalid length");
        }

        int x_rel, y_rel;
        double weight;
        double local_intensity = 0.0;
        auto gaussian_weights_it = gaussian_weights.begin();

        // Although we iterate over a square the gaussian weights
        // are zero outside of the circle with radius "radius" so
        // only sites inside that circle contribute
        for (int i = -radius; i < radius + 1; i++) {
            for (int j = -radius; j < radius + 1; j++) {
                y_rel = math_utils::mod(y + i, height);
                x_rel = math_utils::mod(x + j, width);

                weight = *gaussian_weights_it;

                if constexpr (std::is_same<T, byte>::value) {
                    local_intensity += weight * arr[(y_rel * width) + x_rel];
                }
                else if constexpr (std::is_same<T, cufftDoubleComplex>::value) {
                    local_intensity += weight * math_utils::intensity(arr[(y_rel * width) + x_rel]);
                }
                else {
                    static_assert(false, "get_local_intensity: Invalid type");
                }
                std::advance(gaussian_weights_it, 1);
            }
        }
        return local_intensity;
    }

    std::vector<double> calculate_gaussian_weights(
        int local_intensity_radius, double sigma
    ) const noexcept;

    // Related to camera feedback

    unsigned int camera_px_h, camera_px_v;

    // Generate lattice geometry

    void generate_rectangular_lattice(
        void
    ) noexcept;

    void generate_triangular_lattice(
        void
    ) noexcept;

    void generate_honeycomb_lattice(
        void
    ) noexcept;

    void generate_kagome_lattice(
        void
    );

    void generate_test_array(
        void
    );


};
