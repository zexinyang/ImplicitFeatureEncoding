#include "implicit_feature_encoding.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
ImplicitFeatureEncoding::canonicalizeSampleSphere(const std::vector<uint32_t>& nbrs_query,
                                                  int ptid_query,
                                                  Cloud3D::Ptr& cloud_canonical_sample_sphere) {
    const auto& neighbor_size = nbrs_query.size();
    const auto& pt_query = cloud_input_->points[ptid_query];
    Eigen::MatrixXf cloud_local_2d(neighbor_size, 2); // query point's neighborhood
    uint ptid_highest; // the anchor point
    float z_highest = -static_cast<float>(DBL_MAX);
    for (int i_nbr = 0; i_nbr < neighbor_size; ++i_nbr) {
        // centralize the local area (i.e., i-th point's neighbors)
        cloud_local_2d(i_nbr, 0) = cloud_input_->points[nbrs_query[i_nbr]].x - pt_query.x;
        cloud_local_2d(i_nbr, 1) = cloud_input_->points[nbrs_query[i_nbr]].y - pt_query.y;
        // calculate the local highest point as anchor
        if (cloud_input_->points[nbrs_query[i_nbr]].z > z_highest) {
            ptid_highest = nbrs_query[i_nbr];
            z_highest = cloud_input_->points[nbrs_query[i_nbr]].z;
        }
    }
    Eigen::Vector2f anchor(cloud_input_->points[ptid_highest].x - pt_query.x,
                           cloud_input_->points[ptid_highest].y - pt_query.y);

    // 2D SVD based canonicalization
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(cloud_local_2d, Eigen::ComputeThinV);
    Eigen::Matrix2f v_2d = svd.matrixV();
    // flip new axes so that they point to the anchor
    for (int idrow = 0; idrow < 2; ++idrow) {
        if (anchor.dot(v_2d.col(idrow)) < 0)
            v_2d.col(idrow) *= -1;
    }
    Eigen::Matrix3f v_3d = Eigen::Matrix3f::Identity();
    v_3d.topLeftCorner(2, 2) = v_2d;

    // transform sample points to canonical pose
    // Note that pcl::transformPointCloud() CANNOT be used as it only applies an SO3 or SE3 transform.
    cloud_canonical_sample_sphere->resize(cloud_sample_sphere_->size());
    for (int knlid = 0; knlid < cloud_sample_sphere_->size(); ++knlid) {
        auto pt_vector = cloud_sample_sphere_->points[knlid].getVector3fMap().transpose(); // (1 * 3)
        cloud_canonical_sample_sphere->points[knlid].getVector3fMap() = pt_vector * v_3d.transpose();
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
ImplicitFeatureEncoding::calculateImplicitFeatures(std::vector<std::vector<float>>& implicit_features) {
    // create an octree for neighbor searching
    unibn::Octree<Point3D> octree;
    octree.initialize(*cloud_input_);
    const auto& cloud_size = cloud_input_->size();
    std::vector<std::vector<uint32_t>> neighbors(cloud_size);
    implicit_features.resize(cloud_size);
    // create a kd-tree for nearest neighbor searching,
    // as the octree-based nearest neighbor searching may return empty result (probably a bug)
    pcl::KdTreeFLANN<Point3D> kdtree;
    kdtree.setInputCloud(cloud_input_);
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        // query neighbors
#if defined(_OPENMP)
#pragma omp for
#endif
        for (int i = 0; i < cloud_size; ++i)
            octree.radiusNeighbors<unibn::L2Distance<Point3D>>(
                    cloud_input_->points[i], neighbor_radius_, neighbors[i]);

        // point-wise implicit feature embedding
#if defined(_OPENMP)
#pragma omp for
#endif
        for (int i = 0; i < cloud_size; ++i) {
            // local canonicalization for the sample sphere
            Cloud3D::Ptr cloud_canonical_sample_sphere(new Cloud3D);
            if (neighbors[i].size() < 4)
                // (noted that i-th point's neighbors include itself)
                pcl::copyPointCloud(*cloud_sample_sphere_, *cloud_canonical_sample_sphere);
            else
                canonicalizeSampleSphere(neighbors[i], i, cloud_canonical_sample_sphere);

            // align sample sphere to the query point
            for (auto& knlpt: *cloud_canonical_sample_sphere) {
                knlpt.x += cloud_input_->points[i].x;
                knlpt.y += cloud_input_->points[i].y;
                knlpt.z += cloud_input_->points[i].z;
            }

            // calculate implicit features for the i-th point
            for (auto& knlpt: *cloud_canonical_sample_sphere) {
                std::vector<int> ptid_nearest(1);
                std::vector<float> squared_dist_nearest(1);
                if (kdtree.nearestKSearch(knlpt,
                                          1,
                                          ptid_nearest,
                                          squared_dist_nearest) > 0)
                    implicit_features[i].push_back(std::sqrt(squared_dist_nearest[0]) / sample_radius_);
                else
                    std::cout << "Fail to find any nearest points for the " << i << "-th point. Why?" << std::endl;
            }
        }
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

