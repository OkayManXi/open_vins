#ifndef OV_CORE_TRACK_LARVIO_H
#define OV_CORE_TRACK_LARVIO_H


#include "TrackBase.h"


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

        //IMU state buffer
        std::vector<ImuData> imu_msg_buffer;

        // Take a vector from prev cam frame to curr cam frame
        cv::Matx33f R_Prev2Curr;  
        cv::Matx33d R_cam_imu;

        ImageDataPtr prev_img_ptr;
        ImageDataPtr curr_img_ptr;

        //imu计算位姿差
        void integrateImuData(cv::Matx33f& cam_R_p2c, const std::vector<ImuData>& imu_msg_buffer);

    };

    typedef ImageProcessor::Ptr ImageProcessorPtr;
    typedef ImageProcessor::ConstPtr ImageProcessorConstPtr;

    struct ImuData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ImuData (double t, double wx, double wy, double wz, 
            double ax, double ay, double az) {
            timeStampToSec = t;
            angular_velocity[0] = wx;
            angular_velocity[1] = wy;
            angular_velocity[2] = wz;
            linear_acceleration[0] = ax;
            linear_acceleration[1] = ay;
            linear_acceleration[2] = az;
        }

        ImuData (double t, const Eigen::Vector3d& omg, const Eigen::Vector3d& acc) {
            timeStampToSec = t;
            angular_velocity = omg;
            linear_acceleration = acc;
        }

        double timeStampToSec;
        Eigen::Vector3d angular_velocity;
        Eigen::Vector3d linear_acceleration;
    };

    struct ImgData {
        double timeStampToSec;
        cv::Mat image;
        };

    typedef boost::shared_ptr<ImgData> ImageDataPtr;


}


#endif /* OV_CORE_TRACK_LARVIO_H */