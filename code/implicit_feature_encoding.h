#ifndef IMPLICIT_FEATURE_ENCODING_H
#define IMPLICIT_FEATURE_ENCODING_H

#include <iostream>
// pcl
#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/kdtree/kdtree_flann.h>
// ICRA 2015 octree
#include "octree_unibn.hpp"

typedef pcl::PointXYZ Point3D;
typedef pcl::PointCloud<Point3D> Cloud3D;

class ImplicitFeatureEncoding {
public:
    ImplicitFeatureEncoding() : cloud_input_(new Cloud3D),
                                cloud_sample_sphere_(new Cloud3D),
                                neighbor_radius_(0.15),
                                sample_radius_(0.15),
                                sample_size_(32){
    }

    virtual ~ImplicitFeatureEncoding() = default;

    inline void
    setInputCloud(const Cloud3D::ConstPtr& cloud_input) {
        cloud_input_ = cloud_input;
    }

    inline void
    setSampleSphere(const Cloud3D::ConstPtr& cloud_sample_sphere) {
        cloud_sample_sphere_ = cloud_sample_sphere;
        sample_size_ = cloud_sample_sphere->size();
        // calculate the normalization factor (i.e., distance between the farthest sample position and the sphere center)
        std::vector<float> squared_dist_to_center(sample_size_);
        for (auto i = 0; i < sample_size_; ++i) {
            const auto pt_sample = cloud_sample_sphere->points[i];
            squared_dist_to_center[i] =
                    pt_sample.x * pt_sample.x + pt_sample.y * pt_sample.y + pt_sample.z * pt_sample.z;
        }
        sample_radius_ = std::sqrt(
                *max_element(squared_dist_to_center.begin(), squared_dist_to_center.end()));
    }

    inline void
    setNeighborRadius(const float neighbor_radius) {
        neighbor_radius_ = neighbor_radius;
    }

    inline float
    getNeighborRadius() const {
        return neighbor_radius_;
    }

    inline float
    getSampleRadius() const {
        return sample_radius_;
    }

    inline size_t
    getSampleSize() const {
        return sample_size_;
    }

    void
    calculateImplicitFeatures(std::vector<std::vector<float>>& implicit_features);

private:

    void
    canonicalizeSampleSphere(const std::vector<uint32_t>& nbrs_query,
                             int ptid_query,
                             Cloud3D::Ptr& cloud_canonical_sample_sphere);

    Cloud3D::ConstPtr cloud_input_;
    Cloud3D::ConstPtr cloud_sample_sphere_;
    float neighbor_radius_;
    float sample_radius_;
    size_t sample_size_;
};


#endif //IMPLICIT_FEATURE_ENCODING_H
