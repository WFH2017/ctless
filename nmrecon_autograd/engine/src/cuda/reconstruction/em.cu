#include "cuda/reconstruction/em.h"
#include "cuda/device_arithmetic.h"
#include "cuda/io/utils.h"

#define DEBUG
NMRECON_NAMESPACE_BEGIN
namespace reconstruction
{
    namespace algorithm
    {

        using namespace nmrecon::reconstruction::projector::distance_driven;


        void compute_x_init_value(Scanner &scanner, Image &emap, Image &atten_map,
                                         ProjectionValue &proj_value, GaussianCDR &cdr_model)
        {
            Image identity_image = emap;
            identity_image.fill(1.0f);
            ProjectionValue ax = proj_value;
            // Ax
            cdr_project(scanner, cdr_model, identity_image, atten_map, ax);
            // 
        }
        Image common_recon(Scanner &scanner,
                           Image &emap,
                           Image &atten_map,
                           ProjectionValue &proj_value,
                           int num_iters,
                           int save_interval,
                           const std::string &out_filename,
                           const bool use_cdr = false)
        {
            ProjectionValue proj_ones(scanner.get_total_num_rays(), 1.0f);
            GaussianCDR cdr_model(0.056217, 3.991134, 3.0);
            if (use_cdr)
                cdr_backproject(scanner, cdr_model, emap, atten_map, proj_ones);
            else
                atten_backproject(scanner, emap, atten_map, proj_ones);
            auto emap_data = emap.get_data();
            nmrecon::io::statistic(emap_data, "emap");

            std::string cat = use_cdr ? "cdr" : "atten";
            auto emap_file = out_filename + "_emap_" + cat + ".bin";
            nmrecon::io::save_bin(emap_file.c_str(), emap);

            Image last = emap;
            Image current = emap;
            last.fill(1.0f);
            current.fill(1.0f);
            ProjectionValue ax = proj_value;

            for (int i = 0; i < num_iters; i++)
            {
                if (use_cdr)
                    cdr_project(scanner, cdr_model, current, atten_map, ax);
                else
                    atten_project(scanner, current, atten_map, ax);
                nmrecon::io::statistic(ax, "ax");
                auto project_ratio = nmrecon::common::device_arithmetic::true_div(proj_value, ax);
                current.fill(0.0f);
                if (use_cdr)
                    cdr_backproject(scanner, cdr_model, current, atten_map, project_ratio);
                else
                    atten_backproject(scanner, current, atten_map, project_ratio);

                current = last / emap * current;
                last = current;

                std::cout << " " << i << " th iteration" << std::endl;

// DEBUG
#ifdef DEBUG
                {
                    nmrecon::io::statistic(proj_value, "raw_data");
                    nmrecon::io::statistic(ax, "ax");
                    auto img_data = current.get_data();
                    nmrecon::io::statistic(img_data, "image");
                }
#endif
                if ((save_interval > 0) && (i % save_interval == 0))
                {
                    auto out_file = out_filename + std::to_string(i) + ".bin";
                    nmrecon::io::save_bin(out_file.c_str(), current);
                }
            }
            return current;
        }

        Image osem(Scanner &scanner,
                   Image &emap,
                   Image &atten_map,
                   std::vector<Projection> &projections,
                   int num_iters,
                   int num_subsets,
                   int save_interval,
                   const std::string &out_filename,
                   const bool use_cdr,
                   GaussianCDR &cdr_model)
        {
            auto collimator = scanner.get_collimator();
            auto det_coordinate = collimator.get_coordinate_prototype();
            // auto traj_list = scanner.get_trajectory_list();

            Image last = emap;
            Image current = emap;
            last.fill(1.0f);
            current.fill(1.0f);

            // create projection in subset format.
            Projection projection, scatter_raw;
            if (projections.size() == 1)
            {
                projection = projections[0];
            }
            else if (projections.size() > 1)
            {
                // use the first projection as the main projection.
                projection = projections[0];
                // use the rest of the projections as scatter raw.
                scatter_raw = projections[1];
            }
            else
            {
                throw std::runtime_error("No valid projection provided for OSEM reconstruction.");
            }
            auto subset_projection_list = projection.get_sub_projections(num_subsets);
            num_subsets = subset_projection_list.size();
            
            auto subset_scatter_list = scatter_raw.get_sub_projections(num_subsets);

            // compute emap in subset format.
            host_vector<Image> subset_emap_list(num_subsets, emap);
            for (int s = 0; s < num_subsets; s++)
            {
                Image subset_emap = emap;
                subset_emap.fill(0.0f);
                auto sub_projection = subset_projection_list[s];
                Projection subset_ones = {sub_projection.get_trajectory_list(), sub_projection.get_detector_size(), DeviceVector(sub_projection.get_data().size(), 1.0f)};
                if (use_cdr)
                    generic_cdr_backproject(subset_emap, atten_map, det_coordinate, subset_ones, cdr_model);
                else
                    generic_backproject(subset_emap, atten_map, det_coordinate, subset_ones);
                subset_emap_list[s] = subset_emap;
                auto emap_file = out_filename + "_emap_" + std::to_string(s) + ".bin";
                // nmrecon::io::save_bin(emap_file.c_str(), subset_emap);
                auto img_data = subset_emap.get_data();
                nmrecon::io::statistic(img_data, "emap");
            }

            // iterative reconstruction on iters and subsets
            for (int i = 0; i < num_iters; i++)
            {
                for (int s = 0; s < num_subsets; s++)
                {
                    printf(" %d th iteration, %d th subset\n", i, s);

                    auto sub_projection = subset_projection_list[s];
                    auto sub_scatter = subset_scatter_list[s];
                    auto subset_ax = Projection{sub_projection.get_trajectory_list(), sub_projection.get_detector_size(), DeviceVector(sub_projection.get_data().size(), 0.0f)};
                    if (use_cdr)
                        generic_cdr_project(current, atten_map, det_coordinate, subset_ax, cdr_model);
                    else
                        generic_project(current, atten_map, det_coordinate, subset_ax);
                    // auto current_data = current.get_data();
                    // auto sub_sum = subset_ax+ sub_scatter;
                    // auto ratio = nmrecon::common::device_arithmetic::true_div(sub_projection.get_data(), sub_sum.get_data());
                    auto projection_ratio = sub_projection/(subset_ax + sub_scatter);
                    // set the backproject result to zero.
                    current.fill(0.0f);
                    if (use_cdr)
                        generic_cdr_backproject(current, atten_map, det_coordinate, projection_ratio, cdr_model);
                    else
                        generic_backproject(current, atten_map, det_coordinate, projection_ratio);
                    
                    // need to control the scale of the division result.
                    auto current_sub_projection_data = sub_projection.get_data();
                    auto current_sub_ax_data = subset_ax.get_data();
                    auto current_sub_scatter_data = sub_scatter.get_data();
                    nmrecon::io::statistic(current_sub_projection_data, "projection_iter_" + std::to_string(i) + "_sub_" + std::to_string(s));
                    nmrecon::io::statistic(current_sub_ax_data, "ax_iter_" + std::to_string(i) + "_sub_" + std::to_string(s));
                    // statistic(ratio, "ratio_iter_" + std::to_string(i) + "_sub_" + std::to_string(s));
                    nmrecon::io::statistic(current_sub_scatter_data, "scatter_iter_" + std::to_string(i) + "_sub_" + std::to_string(s));
                    auto current_data = current.get_data();
                    nmrecon::io::statistic(current_data, "backproject");

                    current = last / subset_emap_list[s] * current;
                    last = current;
                    auto img_data = current.get_data();
                    nmrecon::io::statistic(img_data, "image_iter_" + std::to_string(i) + "_sub_" + std::to_string(s));
                    // auto out_file = out_filename + std::to_string(i) + "_" + std::to_string(s) + ".bin";
                    // save_bin(out_file.c_str(), current);
                    
                    

                }
                if ((save_interval > 0) && (i % save_interval == 0))
                {
                    auto out_file = out_filename + '_'+std::to_string(i) + ".bin";
                    nmrecon::io::save_bin(out_file.c_str(), current);
                }
            }
            return current;
        }

        Image ramla(Scanner &scanner,
                    Image &emap,
                    Image &atten_map,
                    std::vector<Projection> &projections,
                    int num_iters,
                    int num_subsets,
                    int save_interval,
                    const std::string &out_filename,
                    const bool use_cdr,
                    GaussianCDR &cdr_model)
        {
            precision_t alpha = 0.9;
            precision_t beta = 4;

            auto collimator = scanner.get_collimator();
            auto det_coordinate = collimator.get_coordinate_prototype();


            Image last = emap;
            Image current = emap;
            last.fill(0.01f);
            current.fill(0.01f);

            // create projection in subset format.
                        // create projection in subset format.
            Projection projection, scatter_raw;
            if (projections.size() == 1)
            {
                projection = projections[0];
            }
            else if (projections.size() > 1)
            {
                // use the first projection as the main projection.
                projection = projections[0];
                // use the rest of the projections as scatter raw.
                scatter_raw = projections[1];
            }
            else
            {
                throw std::runtime_error("No valid projection provided for OSEM reconstruction.");
            }

            auto subset_projection_list = projection.get_sub_projections(num_subsets);
            num_subsets = subset_projection_list.size();

            // compute emap in subset format.
            host_vector<Image> subset_emap_list(num_subsets, emap);
            host_vector<precision_t> max_emap_values(num_subsets);
            for (int s = 0; s < num_subsets; s++)
            {
                Image subset_emap = emap;
                subset_emap.fill(0.0f);
                auto sub_projection = subset_projection_list[s];
                Projection subset_ones = {sub_projection.get_trajectory_list(), sub_projection.get_detector_size(), DeviceVector(sub_projection.get_data().size(), 1.0f)};
                if (use_cdr)
                    generic_cdr_backproject(subset_emap, atten_map, det_coordinate, subset_ones, cdr_model);
                else
                    generic_backproject(subset_emap, atten_map, det_coordinate, subset_ones);
                subset_emap_list[s] = subset_emap;
                auto emap_file = out_filename + "_emap_" + std::to_string(s) + ".bin";
                nmrecon::io::save_bin(emap_file.c_str(), subset_emap);
                auto emap_data = subset_emap.get_data();
                nmrecon::io::statistic(emap_data, "emap");
                // 统计最大值
                thrust::device_vector<precision_t>::iterator max_it = thrust::max_element(emap_data.begin(), emap_data.end());
                auto max_value = *max_it;
                max_emap_values.push_back(max_value);
            }
            auto max_emap_value = *std::max_element(max_emap_values.begin(), max_emap_values.end());

            // 计算
            for (int i = 0; i < num_iters; i++)
            {

                precision_t lambda_i = alpha / (i + beta) / max_emap_value;

                for (int s = 0; s < num_subsets; s++)
                {
                    auto sub_projection = subset_projection_list[s];
                    Projection subset_ax =  {sub_projection.get_trajectory_list(), sub_projection.get_detector_size(), DeviceVector(sub_projection.get_data().size(), 0.0f)};
                    if (use_cdr)
                        generic_cdr_project(current, atten_map, det_coordinate, subset_ax, cdr_model);
                    else
                        generic_project(current, atten_map, det_coordinate, subset_ax);

                    // auto current_data = current.get_data();
                    auto ratio = nmrecon::common::device_arithmetic::true_div(sub_projection.get_data(), subset_ax.get_data());
                    auto projection_ratio = Projection{sub_projection.get_trajectory_list(), sub_projection.get_detector_size(), ratio};
                    
                    current.fill(0.0f);
                    if (use_cdr)
                        generic_cdr_backproject(current, atten_map, det_coordinate, projection_ratio, cdr_model);
                    else
                        generic_backproject(current, atten_map, det_coordinate, projection_ratio);

                    current = last + last * (current * lambda_i - subset_emap_list[s] * lambda_i);

                    last = current;
                    auto out_file = out_filename + std::to_string(i) + "_" + std::to_string(s) + ".bin";
                    // save_bin(out_file.c_str(), current);

                    auto img_data = current.get_data();
                    printf(" %d th iteration, %d th subset\n", i, s);

                    auto current_sub_projection_data = sub_projection.get_data();
                    auto current_sub_ax_data = sub_projection.get_data();
                    nmrecon::io::statistic(current_sub_projection_data, "projection_iter_" + std::to_string(i) + "_sub_" + std::to_string(s));
                    nmrecon::io::statistic(current_sub_ax_data, "ax_iter_" + std::to_string(i) + "_sub_" + std::to_string(s));
                    nmrecon::io::statistic(ratio, "ratio_iter_" + std::to_string(i) + "_sub_" + std::to_string(s));
                    nmrecon::io::statistic(img_data, "image_iter_" + std::to_string(i) + "_sub_" + std::to_string(s));
                }
                if ((save_interval > 0) && (i % save_interval == 0))
                {
                    auto out_file = out_filename + std::to_string(i) + ".bin";
                    nmrecon::io::save_bin(out_file.c_str(), current);
                }
            }
            return current;
        }

        Image mbsrem(Scanner &scanner,
                     Image &emap,
                     Image &atten_map,
                     Projection &projection,
                     int num_iters,
                     int num_subsets,
                     int save_interval,
                     const std::string &out_filename,
                     const bool use_cdr,
                     GaussianCDR &cdr_model)
        {
            Image current = emap;
            return current;

        }
    }
}
NMRECON_NAMESPACE_END
