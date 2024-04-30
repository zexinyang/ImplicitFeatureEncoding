/*
 * Software License Agreement (Apache License)
 *
 *  Copyright (C) 2022, Zexin Yang (zexinyang@gzpi.com.cn).
 *  All rights reserved.
 *
 *  This file is part of ImplicitFeatureEncoding (https://github.com/zexinyang/ImplicitFeatureEncoding),
 *  which implements the method described in the following paper:
 *  -----------------------------------------------------------------------------------------------------------
 *  Enriching Point Clouds with Implicit Representations for 3D Classification and Segmentation.
 *  Zexin Yang, Qin Ye, Jantien Stoter, and Liangliang Nan.
 *  Remote Sensing. 15(1), 61, 2023.
 *  -----------------------------------------------------------------------------------------------------------
 *  We kindly ask you to cite the above paper if you use (part of) the code or ideas in your academic work.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

#include <string>
#include <iostream>
#include <filesystem>
#include <pcl/io/ply_io.h>
#include "implicit_feature_encoding.h"
#include "happly.h"

void
loadCloudFromPLYFile(const std::string& ply_filename, Cloud3D::Ptr& cloud) {
    happly::PLYData ply_in(ply_filename);
    std::vector<double> x = ply_in.getElement("vertex").getProperty<double>("x");
    std::vector<double> y = ply_in.getElement("vertex").getProperty<double>("y");
    std::vector<double> z = ply_in.getElement("vertex").getProperty<double>("z");
    cloud->resize(x.size());
    for (int i_pt = 0; i_pt < x.size(); ++i_pt) {
        cloud->points[i_pt].x = static_cast<float>(x[i_pt]);
        cloud->points[i_pt].y = static_cast<float>(y[i_pt]);
        cloud->points[i_pt].z = static_cast<float>(z[i_pt]);
    }
}

void
saveAugmentedPointCloud(const std::string& file_name,
                        const Cloud3D::Ptr& cloud_input,
                        const std::vector <std::vector<float>>& implicits) {
    const auto& cloud_size = cloud_input->size();
    const auto& kernel_size = implicits.front().size();

    // Restructure the data as dimension-wise stacked vectors for saving
    std::vector <std::vector<float>> dimwise_cloud(3);
    for (int i_dim = 0; i_dim < 3; ++i_dim)
        dimwise_cloud[i_dim].resize(cloud_size);
    for (int i_pt = 0; i_pt < cloud_size; ++i_pt) {
        dimwise_cloud[0][i_pt] = cloud_input->points[i_pt].x;
        dimwise_cloud[1][i_pt] = cloud_input->points[i_pt].y;
        dimwise_cloud[2][i_pt] = cloud_input->points[i_pt].z;
    }

    std::vector <std::vector<float>> kernelwise_implicits(kernel_size);
    for (size_t i_feat = 0; i_feat < kernel_size; ++i_feat) {
        kernelwise_implicits[i_feat].resize(cloud_size);
        for (size_t i_pt = 0; i_pt < cloud_size; ++i_pt)
            kernelwise_implicits[i_feat][i_pt] = implicits[i_pt][i_feat];
    }

    // Create happly ply object
    happly::PLYData ply_out;
    ply_out.addElement("vertex", cloud_size);
    ply_out.getElement("vertex").addProperty<float>("x", dimwise_cloud[0]);
    ply_out.getElement("vertex").addProperty<float>("y", dimwise_cloud[1]);
    ply_out.getElement("vertex").addProperty<float>("z", dimwise_cloud[2]);
    for (int i = 0; i < kernel_size; ++i) {
        std::string feature_name = "d" + std::to_string(i + 1);
        ply_out.getElement("vertex").addProperty<float>(feature_name, kernelwise_implicits[i]);
    }

    // Write the object to file
    ply_out.write(file_name, happly::DataFormat::Binary);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv) {
#if defined(_OPENMP)
    std::cout << "[PARALLEL PROCESSING USING " << omp_get_max_threads() << " THREADS] \n" << std::endl;
#else
    std::cout << "[NON-PARALLEL PROCESSING] \n" << std::endl;
#endif

    /* Set up input parameters */
    std::string path_input = std::string(argv[1]);
    std::string file_sample_sphere = std::string(argv[2]);
    float neighbor_radius = std::atof(argv[3]);
    bool fix_z_axis = std::stoi(argv[4]) != 0;
    std::string path_output = std::string(argv[5]);

    if (!std::filesystem::exists(path_output))
        std::filesystem::create_directory(path_output);

    /* Get a list of filenames in the given directory */
    std::vector <std::string> files;
    for (auto i = std::filesystem::directory_iterator(path_input);
         i != std::filesystem::directory_iterator(); ++i) {
        if (i->path().extension() == ".ply")
            files.push_back(i->path().filename().string());
        else
            std::cerr << i->path().filename().string() << " is not a ply file and thus will be skipped." << std::endl;
    }

    /* Load sample sphere */
    Cloud3D::Ptr cloud_sample_sphere(new Cloud3D);
    pcl::io::loadPLYFile<Point3D>(file_sample_sphere, *cloud_sample_sphere);

    ImplicitFeatureEncoding ife;
    ife.setSampleSphere(cloud_sample_sphere);
    ife.setNeighborRadius(neighbor_radius);
    ife.fixZAxis(fix_z_axis);
    std::cout << "CALCULATION SETTINGS" << std::endl;
    std::cout << "  (1) radius for neighbor searching: " << ife.getNeighborRadius() << " meter(s)" << std::endl;
    std::cout << "  (2) radius of sample sphere: " << ife.getSampleRadius() << " meter(s)" << std::endl;
    std::cout << "  (3) dimension of implicits: " << ife.getSampleSize() << std::endl;

    /* Calculate implicits for each file */
    std::cout << "\nSTART BATCH PROCESSING" << std::endl;
    int id_file = 1;
    const auto& file_num = files.size();
    double time_calculation = 0.0, time_total = 0.0;
    size_t dataset_size = 0;
    for (auto& file: files) {
        std::cout << "\n[" << id_file << "/" << file_num << "] "
                  << "Processing " << file << " ..." << std::endl;
        std::string file_output;
        file_output = path_output + "/" + file.substr(0, file.length() - 4) + ".ply";
        if (std::filesystem::exists(file_output)) {
            std::cout << "  Implicit file already exists. Skipping..." << std::endl;
            id_file++;
            continue;
        }

        double tic, toc;
        // Load raw point cloud
        tic = omp_get_wtime();
        std::string file_input = std::string(argv[1]) + "/" + file;
        Cloud3D::Ptr cloud_input(new Cloud3D);
        loadCloudFromPLYFile(file_input, cloud_input);
        size_t cloud_size = cloud_input->size();
        toc = omp_get_wtime();
        std::cout << "  (1) Load " << cloud_size << " points in " << toc - tic << " seconds." << std::endl;
        dataset_size += cloud_size;

        // Calculate implicits
        tic = omp_get_wtime();
        std::vector <std::vector<float>> implicit_features;
        ife.setInputCloud(cloud_input);
        ife.calculateImplicitFeatures(implicit_features);

        toc = omp_get_wtime();
        std::cout << "  (2) Calculate implicit features in " << toc - tic << " seconds." << std::endl;
        time_calculation += toc - tic;
        time_total += toc - tic;

        // Write augmented point cloud
        tic = omp_get_wtime();
        saveAugmentedPointCloud(file_output, cloud_input, implicit_features);
        toc = omp_get_wtime();
        std::cout << "  (3) Save augmented point cloud in " << toc - tic << " seconds." << std::endl;
        time_total += toc - tic;

        id_file++;
    }

    std::cout << "\nEND BATCH PROCESSING" << std::endl;
    std::cout << "  (1) Number of points of the entire dataset: " << dataset_size << std::endl;
    std::cout << "  (2) Time for implicit calculation: ";
    if (time_calculation > 60)
        std::cout << time_calculation / 60 << " minutes." << std::endl;
    else
        std::cout << time_calculation << " seconds." << std::endl;
    std::cout << "  (3) Total time (calculation + saving): ";
    if (time_total > 60)
        std::cout << time_total / 60 << " minutes." << std::endl;
    else
        std::cout << time_total << " seconds." << std::endl;

    return 0;
}