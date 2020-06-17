#pragma once
#include <string>
#include <fstream>
#include <filesystem>

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
    explicit Parameters();

    // Delete copy and copy assignment
    // I have no idea why but this generates in error when a parameter instance
    // is constructed in a cu file but there's no error in a cpp file
    // Parameters(const Parameters&) = delete;
    // Parameters& operator=(const Parameters&) = delete;

    unsigned int        get_slm_px_x() const;
    unsigned int        get_slm_px_y() const;
    double              get_sensor_size_x_mm() const;
    double              get_sensor_size_y_mm() const;
    double              get_pixel_size_x_mm() const;
    double              get_pixel_size_y_mm() const;

    unsigned int        get_frame_rate() const;
    unsigned int        get_patch_size_x_px() const;
    unsigned int        get_patch_size_y_px() const;

    unsigned int        get_lut_patch_size_x_px() const;
    unsigned int        get_lut_patch_size_y_px() const;

    unsigned int        get_number_of_lut_patches_x() const;
    unsigned int        get_number_of_lut_patches_y() const;

    unsigned int        get_grating_period_px() const;

    unsigned int        get_horizontal_offset() const;

    byte                get_blazed_grating_max() const;

    std::string         get_camera_id() const;
    unsigned int        get_camera_max_frame_rate() const;
    unsigned int        get_camera_px_x() const;
    unsigned int        get_camera_px_y() const;
    double              get_camera_px_size_um() const;
    std::string         get_exposure_mode() const;
    double              get_exposure_time_us() const;

    std::string         get_serial_port_name() const;
    std::string         get_pd_readout_folder() const;
    std::string         get_camera_image_folder() const;
    double              get_axial_scan_range_lower_um() const;
    double              get_axial_scan_range_upper_um() const;
    double              get_axial_scan_stepsize_um() const;

    double              get_focal_length_mm() const;
    double              get_wavelength_um() const;
    double              get_waist_um() const;
    double              get_beam_waist_x_mm() const;
    double              get_beam_waist_y_mm() const;

    unsigned int        get_number_of_pixels_unpadded() const;
    unsigned int        get_number_of_pixels_padded() const;
    unsigned int        get_block_size() const;
    unsigned int        get_num_blocks_padded() const;
    unsigned int        get_num_blocks_unpadded() const;
    unsigned int        get_num_blocks_slm() const;
    unsigned int        get_first_nonzero_index() const;
    int                 get_random_seed() const;
    std::string         get_array_geometry() const;
    unsigned int        get_num_traps_x() const;
    unsigned int        get_num_traps_y() const;
    double              get_spacing_x_um() const;
    double              get_spacing_y_um() const;
    double              get_radial_shift_x_um() const;
    double              get_radial_shift_y_um() const;
    double              get_axial_shift_um() const;

    bool                get_camera_feedback_enabled() const;
    //size_t              get_number_of_pixels_padded() const;
    unsigned int        get_max_iterations() const;
    unsigned int        get_max_iterations_camera_feedback() const;
    unsigned int        get_fixed_phase_limit_iterations() const;
    double              get_fixed_phase_limit_nonuniformity_percent() const;
    double              get_max_nonuniformity_percent() const;
    double              get_max_nonuniformity_camera_feedback_percent() const;

    double              get_weighting_parameter() const;

    std::string         get_output_folder() const;
    bool                get_save_data() const;

    double              get_layer_separation_um() const;

    private:
    json config;
};
