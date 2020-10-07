#ifndef OV_CORE_TRACK_LARVIO_H
#define OV_CORE_TRACK_LARVIO_H


#include "TrackBase.h"
#include "ORBDescriptor.h"
#include "ImuData.hpp"
#include "ImageData.hpp"


namespace ov_core {


    
    class TrackLARVIO : public TrackBase {

    public:

        /**
         * @brief Public default constructor
         */
        TrackLARVIO() : TrackBase(), threshold(10), grid_x(8), grid_y(5), min_px_dist(30) {}

        /**
         * @brief Public constructor with configuration variables
         * @param numfeats number of features we want want to track (i.e. track 200 points from frame to frame)
         * @param numaruco the max id of the arucotags, so we ensure that we start our non-auroc features above this value
         * @param fast_threshold FAST detection threshold
         * @param gridx size of grid in the x-direction / u-direction
         * @param gridy size of grid in the y-direction / v-direction
         * @param minpxdist features need to be at least this number pixels away from each other
         */
        explicit TrackLARVIO(int numfeats, int numaruco, int fast_threshold, int gridx, int gridy, int minpxdist) :
                 TrackBase(numfeats, numaruco), threshold(fast_threshold), grid_x(gridx), grid_y(gridy), min_px_dist(minpxdist) {}


        /**
         * @brief Process a new monocular image
         * @param timestamp timestamp the new image occurred at
         * @param img new cv:Mat grayscale image
         * @param cam_id the camera id that this new image corresponds too
         */
        void feed_monocular(double timestamp, cv::Mat &img, size_t cam_id) override;

        /**
         * @brief Process new stereo pair of images
         * @param timestamp timestamp this pair occured at (stereo is synchronised)
         * @param img_left first grayscaled image
         * @param img_right second grayscaled image
         * @param cam_id_left first image camera id
         * @param cam_id_right second image camera id
         */
        void feed_stereo(double timestamp, cv::Mat &img_left, cv::Mat &img_right, size_t cam_id_left, size_t cam_id_right) override;

        void feedimu(double timestamp, Eigen::Vector3d wm, Eigen::Vector3d am) override;

        void setcameraintrinsics(std::map<size_t,Eigen::VectorXd> camera_calib,const Eigen::Matrix<double, 4, 1> camex) override;

    protected:

        /**
         * @brief Detects new features in the current image
         * @param img0pyr image we will detect features on (first level of pyramid)
         * @param pts0 vector of currently extracted keypoints in this image
         * @param ids0 vector of feature ids for each currently extracted keypoint
         *
         * Given an image and its currently extracted features, this will try to add new features if needed.
         * Will try to always have the "max_features" being tracked through KLT at each timestep.
         * Passed images should already be grayscaled.
         */
        void perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, std::vector<cv::KeyPoint> &pts0, std::vector<size_t> &ids0);

        /**
         * @brief Detects new features in the current stereo pair
         * @param img0pyr left image we will detect features on (first level of pyramid)
         * @param img1pyr right image we will detect features on (first level of pyramid)
         * @param pts0 left vector of currently extracted keypoints
         * @param pts1 right vector of currently extracted keypoints
         * @param ids0 left vector of feature ids for each currently extracted keypoint
         * @param ids1 right vector of feature ids for each currently extracted keypoint
         *
         * This does the same logic as the perform_detection_monocular() function, but we also enforce stereo contraints.
         * So we detect features in the left image, and then KLT track them onto the right image.
         * If we have valid tracks, then we have both the keypoint on the left and its matching point in the right image.
         * Will try to always have the "max_features" being tracked through KLT at each timestep.
         */
        void perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &pts0,
                                      std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1);

        /**
         * @brief KLT track between two images, and do RANSAC afterwards
         * @param img0pyr starting image pyramid
         * @param img1pyr image pyramid we want to track too
         * @param pts0 starting points
         * @param pts1 points we have tracked
         * @param id0 id of the first camera
         * @param id1 id of the second camera
         * @param mask_out what points had valid tracks
         *
         * This will track features from the first image into the second image.
         * The two point vectors will be of equal size, but the mask_out variable will specify which points are good or bad.
         * If the second vector is non-empty, it will be used as an initial guess of where the keypoints are in the second image.
         */
        void perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &pts0,
                              std::vector<cv::KeyPoint> &pts1, size_t id0, size_t id1, std::vector<uchar> &mask_out);

        // Timing variables
        boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;

        // Parameters for our FAST grid detector
        int threshold;
        int grid_x;
        int grid_y;

        // Minimum pixel distance to be "far away enough" to be a different extracted feature
        int min_px_dist;

        // How many pyramid levels to track on and the window size to reduce by
        int pyr_levels = 3;
        cv::Size win_size = cv::Size(15, 15);

        // Last set of image pyramids
        std::map<size_t, std::vector<cv::Mat>> img_pyramid_last;

    //private:
        //增加的
        //IMU state buffer

        std::vector<ImuData> imu_msg_buffer;

        // Take a vector from prev cam frame to curr cam frame
        cv::Matx33f R_Prev2Curr;  
        cv::Matx33d R_cam_imu;

        // Pyramids for previous and current image
        std::vector<cv::Mat> prev_pyramid_;
        std::vector<cv::Mat> curr_pyramid_;
        typedef unsigned long int FeatureIDType;

        FeatureIDType next_feature_id=0;

        ImageDataPtr prev_img_ptr;
        ImageDataPtr curr_img_ptr;

        int before_tracking;
        int after_tracking;
        int after_ransac;

        // Points for tracking, added by QXC
        std::vector<cv::Point2f> new_pts_;
        std::vector<cv::Point2f> prev_pts_;
        std::vector<cv::Point2f> curr_pts_;
        std::vector<FeatureIDType> pts_ids_;
        std::vector<int> pts_lifetime_;
        std::vector<cv::Point2f> init_pts_;

        // Time of last published image
        double last_pub_time;
        double curr_img_time;
        double prev_img_time;

        cv::Vec4d cam_intrinsics;

        template <typename T>
        void removeUnmarkedElements(
            const std::vector<T>& raw_vec,
            const std::vector<unsigned char>& markers,
            std::vector<T>& refined_vec) {
            if (raw_vec.size() != markers.size()) {
                for (int i = 0; i < raw_vec.size(); ++i)
                refined_vec.push_back(raw_vec[i]);
            return;
            }
            for (int i = 0; i < markers.size(); ++i) {
                if (markers[i] == 0) continue;
                refined_vec.push_back(raw_vec[i]);
            }
            return;
        }

        void undistortPoints(const std::vector<cv::Point2f>& pts_in,
        const cv::Vec4d& intrinsics, const cv::Vec4d& distortion_coeffs,
        std::vector<cv::Point2f>& pts_out, const cv::Matx33d &rectification_matrix = cv::Matx33d::eye(),
        const cv::Vec4d &new_intrinsics = cv::Vec4d(1,1,0,0));

        //imu计算位姿差
        void integrateImuData(cv::Matx33f& cam_R_p2c, const std::vector<ImuData>& imu_msg_buffer);

        void predictFeatureTracking(const std::vector<cv::Point2f>& input_pts,const cv::Matx33f& R_p_c,const cv::Vec4d& intrinsics,std::vector<cv::Point2f>& compenstated_pts);

        bool initializeFirstFrame();

        bool initializeFirstFeatures(const std::vector<ImuData>& imu_msg_buffer);

        void createImagePyramids();

        void trackFeatures();

        void trackNewFeatures();

        void findNewFeaturesToBeTracked();
        

        // Enum type for image state.
        enum eImageState {
            FIRST_IMAGE = 1,
            SECOND_IMAGE = 2,
            OTHER_IMAGES = 3
        };
        // Indicate if this is the first or second image message.
        eImageState image_state=FIRST_IMAGE;

        bool bFirstImg;

        // ORB descriptor pointer, added by QXC
        boost::shared_ptr<ORBdescriptor> prevORBDescriptor_ptr;
        boost::shared_ptr<ORBdescriptor> currORBDescriptor_ptr;
        std::vector<cv::Mat> vOrbDescriptors;
    };




}


#endif /* OV_CORE_TRACK_LARVIO_H */