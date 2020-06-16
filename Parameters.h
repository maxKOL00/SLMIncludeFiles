#pragma once
#include <string>
#include <fstream>

//#include "basic_fileIO.h"

#include "nlohmann/json.hpp"

using json = nlohmann::json;
typedef unsigned char byte;

struct ParameterError : public std::runtime_error {
    std::string msg;
    ParameterError(const char* message) throw() : std::runtime_error(message) {
        msg = std::string(message);
    }
    const char* what() const throw() {
        return msg.c_str();
    }
};


class Parameters {
    public:
        explicit Parameters(void);

        // Delete copy and copy assignment
        // I have no idea why but this generates in error when a parameter instance
        // is constructed in a cu file but there's no error in a cpp file
        // Parameters(const Parameters&) = delete;
        // Parameters& operator=(const Parameters&) = delete;

        size_t              get_slm_px_x(void) const;
        size_t              get_slm_px_y(void) const;
        double              get_sensor_size_x_mm(void) const;
        double              get_sensor_size_y_mm(void) const;
        double              get_pixel_size_x_mm(void) const;
        double              get_pixel_size_y_mm(void) const;

        size_t              get_frame_rate(void) const;
        size_t              get_patch_size_x_px(void) const;
        size_t              get_patch_size_y_px(void) const;

        size_t              get_lut_patch_size_x_px(void) const;
        size_t              get_lut_patch_size_y_px(void) const;

        size_t              get_number_of_lut_patches_x(void) const;
        size_t              get_number_of_lut_patches_y(void) const;

        size_t              get_grating_period_px(void) const;
        size_t              get_binary_grating_width_px(void) const;

        size_t              get_horizontal_offset(void) const;

        byte                get_blazed_grating_max(void) const;

        std::string         get_camera_id(void) const;
        size_t              get_camera_max_frame_rate(void) const;
        size_t              get_camera_px_x(void) const;
        size_t              get_camera_px_y(void) const;
        double              get_camera_px_size_um(void) const;
        std::string         get_exposure_mode(void) const;
        double              get_exposure_time_us(void) const;

        std::string         get_serial_port_name(void) const;
        std::string         get_pd_readout_folder(void) const;
        std::string         get_camera_image_folder(void) const;
        double              get_axial_scan_range_lower_um(void) const;
        double              get_axial_scan_range_upper_um(void) const;
        double              get_axial_scan_stepsize_um(void) const;

        double              get_focal_length_mm(void) const;
        double              get_wavelength_um(void) const;
        double              get_waist_um(void) const;
        double              get_beam_waist_x_mm(void) const;
        double              get_beam_waist_y_mm(void) const;

        std::string         get_array_geometry(void) const;
        size_t              get_num_traps_x(void) const;
        size_t              get_num_traps_y(void) const;
        double              get_spacing_x_um(void) const;
        double              get_spacing_y_um(void) const;
        double              get_radial_shift_x_um(void) const;
        double              get_radial_shift_y_um(void) const;
        double              get_axial_shift_um(void) const;

        bool                get_camera_feedback_enabled(void) const;
        size_t              get_number_of_pixels_padded(void) const;
        size_t              get_max_iterations(void) const;
        size_t              get_max_iterations_camera_feedback(void) const;
        size_t              get_fixed_phase_limit_iterations(void) const;
        double              get_fixed_phase_limit_nonuniformity_percent(void) const;
        double              get_max_nonuniformity_percent(void) const;
        double              get_max_nonuniformity_camera_feedback_percent(void) const;
        
        double              get_weighting_parameter(void) const;

        std::string         get_output_folder(void) const;
        bool                get_save_data(void) const;

        double              get_layer_separation_um(void) const;

    private:
        json config;
};
