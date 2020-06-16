#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // transform
#include <iterator>
#include <type_traits>
#include "Constants.h"

#include "statusBox.h"

#include <Windows.h>


#include "Constants.h"
#include "Parameters.h"
#include "Tweezer.h"
#include "basic_fileIO.h" // Just for debugging, can be removed later
#include "math_utils.cuh"


class TweezerArray {
	public:
		explicit TweezerArray(
            const Parameters& params, statusBox* box
        );

        // This could be noexcept because the mask creation already checks if the correct
        // number of peaks is detected
        void update_position_in_camera_image(
            const std::vector<size_t>& sorted_flattened_peak_indices
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
        ) ;

        // In CGHAlgorithm the main WGS loop calculated the weights which are used
        // to update the target amplitude of each tweezer
        void update_target_intensities(
            const std::vector<double>& weights
        ) ;

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
        size_t get_array_size(
            void
        ) const noexcept;
		

    private:
        // 
        size_t num_traps_x, num_traps_y, num_traps;
        double spacing_x_um, spacing_y_um;
        size_t number_of_pixels_unpadded, number_of_pixels_padded;

        // Optical parameters and derived quantities
        double focal_length_um, wavelength_um;

        statusBox* editT;
        // Resolutions in padded Fourier space when delta k
        // is transformed by the scale factor wavelenghth * focal_length
        double delta_x_padded_um;

        // waist_px also declared as double to avoid rounding errors if possible
        double waist_um, waist_px_in_fft, waist_px_in_camera_image;

        // Mean intensity of all local intensities in the tweezer array
        double mean_intensity, mean_amplitude, standard_deviation;

        std::string geometry;

        // Main data structure to operate on
        std::vector<Tweezer> tweezer_vec;

        bool position_in_camera_image_set;

        // get_local_intensity calculates the local intensity of the tweezer spots
        // in both the fft array and the camera image. Because these are different
        // types (cufftDoubleComplex and byte) and they are physically different
        // as the fft array represents the electric field whereas the camera_image
        // just has intensity information, the former needs to be abs-squared 
        // (and we want that to happen at compile time, see below).
        // In c++17 this could easily be implemented with a if constexpr(...)
        // where ... checks the type 
        // (std::is_same<T, byte>::value/std::is_same<T, cufftDoubleComplex>::value).
        // Unfortunately c++17 is not available in cuda 10.2 so one has to come
        // up with something else.
        // I chose to just define an overloaded transform function which solves the
        // problem easily. It is also possible to implement a (more complicated but
        // more general) structure static_if so it can be checked at compile time.
        // In addition to the performance argument static if (or if constexpr) has 
        // the advantage that the branch that is not chosen does not even need to
        // compile, see 
        // https://baptiste-wicht.com/posts/2015/07/simulate-static_if-with-c11c14.html

        inline double transform_func(const cufftDoubleComplex& z) const noexcept{
            return math_utils::intensity(z);
        }

        inline double transform_func(double x) const noexcept {
            return x;
        }

        template<typename T>
        double get_local_intensity(
            const T* arr,
            int x, int y, int width, int height,
            int radius,
            const std::vector<double>& gaussian_weights
        ) const noexcept {
            // Indices are defined as ints so negative values can wrap around 
            // when mod number_of_pixels_padded

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

                    local_intensity += weight * transform_func(arr[(y_rel * width) + x_rel]);

                    std::advance(gaussian_weights_it, 1);
                }
            }
            return local_intensity;
        }

        std::vector<double> calculate_gaussian_weights(
            int local_intensity_radius, double sigma
        ) const noexcept;

        // Related to camera feedback

        size_t camera_px_h, camera_px_v;

        // Generate the target array structure. Information about
        // each site is stored in the Tweezer struct
        const enum Geometry { square, triangular, honeycomb, kagome, test };
        const std::map<std::string, Geometry> geometry_map {
            {"SQUARE", square},
            {"TRIANGULAR", triangular},
            {"HONEYCOMB", honeycomb},
            {"KAGOME", kagome},
            {"TEST", test}
        };

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
