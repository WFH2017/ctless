#include "scanner/spect.h"

NMRECON_NAMESPACE_BEGIN
namespace scanner
{
    SPECT::SPECT(const nmrecon::io::ScanSetting &scan_setting)
    {
        // construct collimator
        precision_t collimator_length = scan_setting.collimator_length;
        auto size = scan_setting.detector_coordinate.size;
        auto spacing = scan_setting.detector_coordinate.spacing;
        auto origin = scan_setting.detector_coordinate.origin;
        auto direction = scan_setting.detector_coordinate.direction;

        Coordinate3D coord;
        coord.set_size(S3(size[0], size[1], size[2]));
        coord.set_spacing(F3(spacing[0], spacing[1], spacing[2]));
        coord.set_origin(F3(origin[0], origin[1], origin[2]));
        ParallelCollimator collimator_(collimator_length, coord);

        // construct projection trajectory
        int num_views = scan_setting.num_views;

        size_t num_trajectories = scan_setting.detector_start_degree.size();
        
        TrajectoryList trajectory_list_;
        for (size_t i = 0; i < num_trajectories; i++)
        {
            std::vector<precision_t> radii_in_trajectory;
            radii_in_trajectory.assign(scan_setting.rotation_radii.begin() + i*num_views, scan_setting.rotation_radii.begin() + (i + 1) * num_views);
            precision_t start_angle = to_radians(scan_setting.detector_start_degree[i]);
            precision_t end_angle = to_radians(scan_setting.detector_end_degree[i]);
            Trajectory traj(radii_in_trajectory, start_angle, end_angle, num_views);
            trajectory_list_.push_back(traj);
        }

        this->collimator = collimator_;
        this->trajectory_list = trajectory_list_;
    }
    
    const Trajectory SPECT::get_trajectory_by_index(const size_t &index) const
    {
        if (index >= trajectory_list.size())
        {
            throw std::out_of_range("index is out of range");
        }
        return trajectory_list[index];
    }

    size_t SPECT::get_total_views() const
    {
        size_t total_views = 0;
        for (size_t i = 0; i < trajectory_list.size(); i++)
        {
            total_views += trajectory_list[i].get_num_views();
        }
        return total_views;
    }

    size_t SPECT::get_total_num_rays() const
    {
        size_t total_num_of_rays = 0;
        for (size_t i = 0; i < trajectory_list.size(); i++)
        {
            total_num_of_rays += trajectory_list[i].get_num_views() * collimator.get_num_rays();
        }
        return total_num_of_rays;
    }

    void SPECT::print() const
    {
        std::cout << "SPECT scanner" << std::endl;
        collimator.print();
        std::cout << "Trajectory list size: " << trajectory_list.size() << std::endl;
        TrajectoryList trajectory_list_host = trajectory_list;
        for (size_t i = 0; i < trajectory_list_host.size(); i++)
        {
            std::cout << "Trajectory " << i << std::endl;
            auto traj = trajectory_list_host[i];
            traj.print();
        }
    }

    precision_t to_radians(precision_t deg)
    {
        return (precision_t)deg * nmrecon::common::kPi / 180.0;
    }

    SPECT make_scanner(SPECT_Type type)
    {
        switch (type)
        {

        case SPECT_Type::Infinia:
        {
            Coordinate3D coord;
            size_t sy = 64, sz = 64;
            precision_t dy = 6.643699, dz = 6.643699;
            precision_t collimator_length = 35.0;

            coord.set_size(S3(1, sy, sz));
            coord.set_spacing(F3(1, dy, dz));
            coord.set_origin(F3(0.0, (-dy * sy + dy) / 2, (-dz * sz + dz) / 2));
            ParallelCollimator collimator(collimator_length, coord);

            size_t num_views = 60;
            precision_t radius = 360;
            precision_t start_angle1 = to_radians(45.0 + 180);
            precision_t end_angle1 = to_radians(45 + 180 + 177.0);
            Trajectory traj1(radius, start_angle1, end_angle1, num_views);

            TrajectoryList trajectory_list;
            trajectory_list.push_back(traj1);
            return SPECT(collimator, trajectory_list);
        }
        case SPECT_Type::Test1024:
        {

            Coordinate3D coord;
            size_t sy = 1024, sz = 1024;
            precision_t dy = 0.56, dz = 0.56;
            precision_t collimator_length = 35.6;

            coord.set_size(S3(1, sy, sz));
            coord.set_spacing(F3(1, dy, dz));
            coord.set_origin(F3(0.0, (-dy * sy + dy) / 2, (-dz * sz + dz) / 2));
            ParallelCollimator collimator(collimator_length, coord);

            size_t num_views = 120;
            precision_t radius = 150.0;
            precision_t start_angle = to_radians(0.0);
            precision_t end_angle = to_radians(357.0);
            Trajectory traj(radius, start_angle, end_angle, num_views);

            TrajectoryList trajectory_list;
            trajectory_list.push_back(traj);

            return SPECT(collimator, trajectory_list);
        }
        case SPECT_Type::Test512Simu:
        {
            Coordinate3D coord;
            size_t sy = 512, sz = 512;
            precision_t dy = 1.12, dz = 1.12;
            precision_t collimator_length = 35.6;

            coord.set_size(S3(1, sy, sz));
            coord.set_spacing(F3(1, dy, dz));
            coord.set_origin(F3(0.0, (-dy * sy + dy) / 2, (-dz * sz + dz) / 2));
            ParallelCollimator collimator(collimator_length, coord);

            size_t num_views = 120;
            precision_t radius = 150.0;
            precision_t start_angle = to_radians(3.0);
            precision_t end_angle = to_radians(360.0);
            Trajectory traj(radius, start_angle, end_angle, num_views);

            TrajectoryList trajectory_list;
            trajectory_list.push_back(traj);

            return SPECT(collimator, trajectory_list);
        }
        case SPECT_Type::Test256Simu:
        {
            Coordinate3D coord;
            size_t sy = 256, sz = 256;
            precision_t dy = 2.24, dz = 2.24;
            precision_t collimator_length = 35.6;

            coord.set_size(S3(1, sy, sz));
            coord.set_spacing(F3(1, dy, dz));
            coord.set_origin(F3(0.0, (-dy * sy + dy) / 2, (-dz * sz + dz) / 2));
            ParallelCollimator collimator(collimator_length, coord);

            size_t num_views = 120;
            precision_t radius = 200.0;
            precision_t start_angle = to_radians(3.0);
            precision_t end_angle = to_radians(360.0);
            Trajectory traj(radius, start_angle, end_angle, num_views);

            TrajectoryList trajectory_list;
            trajectory_list.push_back(traj);

            return SPECT(collimator, trajectory_list);
        }
        case SPECT_Type::Test256:
        {
            Coordinate3D coord;
            size_t sy = 256, sz = 256;
            precision_t dy = 2.24, dz = 2.24;
            precision_t collimator_length = 35.6;

            coord.set_size(S3(1, sy, sz));
            coord.set_spacing(F3(1, dy, dz));
            coord.set_origin(F3(0.0, (-dy * sy + dy) / 2, (-dz * sz + dz) / 2));
            ParallelCollimator collimator(collimator_length, coord);

            size_t num_views = 120;
            precision_t radius = 150.0;
            precision_t start_angle = to_radians(0.0);
            precision_t end_angle = to_radians(357.0);
            Trajectory traj(radius, start_angle, end_angle, num_views);

            TrajectoryList trajectory_list;
            trajectory_list.push_back(traj);

            return SPECT(collimator, trajectory_list);
        }
        case SPECT_Type::Test128Simu:
        {
            Coordinate3D coord;
            size_t sy = 128, sz = 128;
            precision_t dy = 4.48, dz = 4.48;
            precision_t collimator_length = 35.6;

            coord.set_size(S3(1, sy, sz));
            coord.set_spacing(F3(1, dy, dz));
            coord.set_origin(F3(0.0, (-dy * sy + dy) / 2, (-dz * sz + dz) / 2));
            ParallelCollimator collimator(collimator_length, coord);

            size_t num_views = 120;
            precision_t radius = 150.0;
            precision_t start_angle = to_radians(3.0);
            precision_t end_angle = to_radians(360.0);
            Trajectory traj(radius, start_angle, end_angle, num_views);

            TrajectoryList trajectory_list;
            trajectory_list.push_back(traj);

            return SPECT(collimator, trajectory_list);
        }
        case SPECT_Type::Test150Simu:
        {
            Coordinate3D coord;
            size_t sy = 150, sz = 150;
            precision_t dy = 1.2, dz = 1.2;
            precision_t collimator_length = 35.6;

            coord.set_size(S3(1, sy, sz));
            coord.set_spacing(F3(1, dy, dz));
            coord.set_origin(F3(0.0, (-dy * sy + dy) / 2, (-dz * sz + dz) / 2));
            ParallelCollimator collimator(collimator_length, coord);

            size_t num_views = 60;
            precision_t radius = 100.0;
            precision_t start_angle = to_radians(6.0);
            precision_t end_angle = to_radians(360.0);
            Trajectory traj(radius, start_angle, end_angle, num_views);

            TrajectoryList trajectory_list;
            trajectory_list.push_back(traj);

            return SPECT(collimator, trajectory_list);
        }
        default:
        {
            std::cerr << "Unsupported SPECT type";
            break;
        }
        }
    }

}

NMRECON_NAMESPACE_END