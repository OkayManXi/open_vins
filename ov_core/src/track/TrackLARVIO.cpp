#include "TrackLARVIO.h"


using namespace ov_core;

void TrackLARVIO::findNewFeaturesToBeTracked() {
    const cv::Mat& curr_img = curr_pyramid_[0];

    // Create a mask to avoid redetecting existing features.
    cv::Mat mask(curr_img.rows,curr_img.cols,CV_8UC1,255);
    for (const auto& pt : curr_pts_) {
        // int startRow = round(pt.y) - processor_config.patch_size;
        //round整数
        int startRow = std::round(pt.y) - min_px_dist;
        startRow = (startRow<0) ? 0 : startRow;

        // int endRow = round(pt.y) + processor_config.patch_size;
        int endRow = std::round(pt.y) + min_px_dist;
        endRow = (endRow>curr_img.rows-1) ? curr_img.rows-1 : endRow;

        // int startCol = round(pt.x) - processor_config.patch_size;
        int startCol = std::round(pt.x) - min_px_dist;
        startCol = (startCol<0) ? 0 : startCol;

        // int endCol = round(pt.x) + processor_config.patch_size;
        int endCol = std::round(pt.x) + min_px_dist;
        endCol = (endCol>curr_img.cols-1) ? curr_img.cols-1 : endCol;

        cv::Mat mROI(mask,
                    cv::Rect(startCol,startRow,endCol-startCol+1,endRow-startRow+1));
        mROI.setTo(0);
    }

    // detect new features to be tracked
    std::vector<cv::Point2f>().swap(new_pts_);
    if (num_features-curr_pts_.size() > 0)
        cv::goodFeaturesToTrack(curr_img, new_pts_, 
            num_features-curr_pts_.size(), 0.01, min_px_dist, mask);
}

void TrackLARVIO::trackNewFeatures() {
    // Return if no new features
    int num_new = new_pts_.size();
    if ( num_new<=0 ) {
        // printf("NO NEW FEATURES EXTRACTED IN LAST IMAGE");
        return;
    }
    // else
    //     printf("%d NEW FEATURES EXTRACTED IN LAST IMAGE",num_new);

    // Pridict features in current image
    std::vector<cv::Point2f> curr_pts(new_pts_.size());
    predictFeatureTracking(
        new_pts_, R_Prev2Curr, cam_intrinsics, curr_pts);

    // Using LK optical flow to track feaures
    std::vector<unsigned char> track_inliers(new_pts_.size());
    calcOpticalFlowPyrLK(
        prev_pyramid_, curr_pyramid_,
        new_pts_, curr_pts,
        track_inliers, cv::noArray(),
        win_size,
        pyr_levels,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 15, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // Mark those tracked points out of the image region
    // as untracked.
    for (int i = 0; i < curr_pts.size(); ++i) {   
        if (track_inliers[i] == 0) continue;
        if (curr_pts[i].y < 0 ||
            curr_pts[i].y > curr_img_ptr->image.rows-1 ||
            curr_pts[i].x < 0 ||
            curr_pts[i].x > curr_img_ptr->image.cols-1)
            track_inliers[i] = 0;
    }  

    // Use inliers to do RANSAC and further remove outliers
    std::vector<cv::Point2f> prev_pts_inImg_(0);
    std::vector<cv::Point2f> curr_pts_inImg_(0);
    removeUnmarkedElements(   
            new_pts_, track_inliers, prev_pts_inImg_);
    removeUnmarkedElements(
            curr_pts, track_inliers, curr_pts_inImg_);

    // Return if no new feature was tracked
    int num_tracked = prev_pts_inImg_.size();
    if ( num_tracked<=0 ) {
        // printf("NO NEW FEATURE IN LAST IMAGE WAS TRACKED");
        return;
    }

    // Using reverse LK optical flow tracking to eliminate outliers
    std::vector<unsigned char> reverse_inliers(curr_pts_inImg_.size());
    std::vector<cv::Point2f> prev_pts_cpy(prev_pts_inImg_);
    calcOpticalFlowPyrLK(
        curr_pyramid_, prev_pyramid_, 
        curr_pts_inImg_, prev_pts_cpy,
        reverse_inliers, cv::noArray(),
        win_size,
        pyr_levels,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 15, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    // Mark those tracked points out of the image region
    // as untracked.
    for (int i = 0; i < prev_pts_cpy.size(); ++i) {  
        if (reverse_inliers[i] == 0) continue;
        if (prev_pts_cpy[i].y < 0 ||
            prev_pts_cpy[i].y > prev_pyramid_[0].rows-1 ||
            prev_pts_cpy[i].x < 0 ||
            prev_pts_cpy[i].x > prev_pyramid_[0].cols-1) {
            reverse_inliers[i] = 0;
            continue;
        }
        float dis = cv::norm(prev_pts_cpy[i]-prev_pts_inImg_[i]);
        if (dis > 1)    
            reverse_inliers[i] = 0;
    }
    // Remove outliers
    std::vector<cv::Point2f> prev_pts_inImg(0);
    std::vector<cv::Point2f> curr_pts_inImg(0);
    removeUnmarkedElements(    
            prev_pts_inImg_, reverse_inliers, prev_pts_inImg);
    removeUnmarkedElements(
            curr_pts_inImg_, reverse_inliers, curr_pts_inImg);
    // Return if no new feature was tracked
    num_tracked = prev_pts_inImg.size();
    if ( num_tracked<=0 ) {
        // printf("NO NEW FEATURE IN LAST IMAGE WAS TRACKED");
        return;
    }

    // Mark as outliers if descriptor distance is too large
    std::vector<int> levels(prev_pts_inImg.size(), 0);
    cv::Mat prevDescriptors, currDescriptors;
    if (!prevORBDescriptor_ptr->computeDescriptors(prev_pts_inImg, levels, prevDescriptors) ||
        !currORBDescriptor_ptr->computeDescriptors(curr_pts_inImg, levels, currDescriptors)) {
        cerr << "error happen while compute descriptors" << endl;
        return;
    }
    std::vector<int> vDis;
    for (int j = 0; j < prevDescriptors.rows; ++j) {
        int dis = ORBdescriptor::computeDescriptorDistance(
                prevDescriptors.row(j), currDescriptors.row(j));
        vDis.push_back(dis);
    }
    std::vector<unsigned char> desc_inliers(prev_pts_inImg.size(), 0);
    std::vector<cv::Mat> desc_new(0);
    for (int i = 0; i < prev_pts_inImg.size(); i++) {
        if (vDis[i]<=58) {  
            desc_inliers[i] = 1;
            desc_new.push_back(prevDescriptors.row(i));
        }
    }

    // Remove outliers
    std::vector<cv::Point2f> prev_pts_inlier(0);
    std::vector<cv::Point2f> curr_pts_inlier(0);
    removeUnmarkedElements(    
            prev_pts_inImg, desc_inliers, prev_pts_inlier);
    removeUnmarkedElements(
            curr_pts_inImg, desc_inliers, curr_pts_inlier);

    // Return if not enough inliers
    if ( prev_pts_inlier.size()<20 ){
        // printf("NO NEW FEATURE IN LAST IMAGE WAS TRACKED");
        return;
    }

    // Undistort inliers
    std::vector<cv::Point2f> prev_unpts_inlier(prev_pts_inlier.size());
    std::vector<cv::Point2f> curr_unpts_inlier(curr_pts_inlier.size());
    undistortPoints(
            prev_pts_inlier, cam_intrinsics,
            camera_d_OPENCV.at(0), prev_unpts_inlier, 
            cv::Matx33d::eye(), cam_intrinsics);
    undistortPoints(
            curr_pts_inlier, cam_intrinsics,
            camera_d_OPENCV.at(0), curr_unpts_inlier, 
            cv::Matx33d::eye(), cam_intrinsics);

    std::vector<unsigned char> ransac_inliers;

    findFundamentalMat(
            prev_unpts_inlier, curr_unpts_inlier,
            cv::FM_RANSAC, 1.0, 0.99, ransac_inliers);

    std::vector<cv::Point2f> prev_pts_matched(0);
    std::vector<cv::Point2f> curr_pts_matched(0);
    std::vector<cv::Mat> prev_desc_matched(0);
    removeUnmarkedElements(
            prev_pts_inlier, ransac_inliers, prev_pts_matched);
    removeUnmarkedElements(
            curr_pts_inlier, ransac_inliers, curr_pts_matched);
    removeUnmarkedElements(
            desc_new, ransac_inliers, prev_desc_matched);

    // Return if no new feature was tracked
    int num_ransac = curr_pts_matched.size();
    if ( num_ransac<=0 ) {
        // printf("NO NEW FEATURE IN LAST IMAGE WAS TRACKED");
        return;
    }

    // Fill initialized features into init_pts_, curr_pts_, 
    // and set their ids and lifetime
    for (int i = 0; i < prev_pts_matched.size(); ++i) {
        prev_pts_.push_back(prev_pts_matched[i]);
        curr_pts_.push_back(curr_pts_matched[i]);
        pts_ids_.push_back(next_feature_id++);
        pts_lifetime_.push_back(2);
        init_pts_.push_back(prev_pts_matched[i]);
        vOrbDescriptors.push_back(prev_desc_matched[i]);
    }

    // Clear new_pts_
    std::vector<cv::Point2f>().swap(new_pts_);
}

void TrackLARVIO::trackFeatures() {
    // Number of the features before tracking.
    before_tracking = prev_pts_.size();

    // Abort tracking if there is no features in
    // the previous frame.
    if (0 == before_tracking) {
        printf("No feature in prev img !\n");
        return;
    }

    // Pridict features in current image
    std::vector<cv::Point2f> curr_points(prev_pts_.size());
    //上一帧特征、前后帧相机旋转矩阵、相机内参、当前需要检测的特征的vector
    //currpoints为上一帧的特征点坐标通过相机旋转矩阵映射到当前帧中
    predictFeatureTracking(
        prev_pts_, R_Prev2Curr, cam_intrinsics, curr_points);

    // Using LK optical flow to track feaures（光流法）
    //trackinliers追踪局内点
    std::vector<unsigned char> track_inliers(prev_pts_.size());
    calcOpticalFlowPyrLK(
        prev_pyramid_, curr_pyramid_,
        prev_pts_, curr_points,
        track_inliers, cv::noArray(),
        win_size,
        pyr_levels,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 15, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // Mark those tracked points out of the image region
    // as untracked.
    for (int i = 0; i < curr_points.size(); ++i) {   
        if (track_inliers[i] == 0) continue;
        if (curr_points[i].y < 0 ||
            curr_points[i].y > curr_img_ptr->image.rows-1 ||
            curr_points[i].x < 0 ||
            curr_points[i].x > curr_img_ptr->image.cols-1)
            track_inliers[i] = 0;
    }  

    // Collect the tracked points.
    //usign longlong int
    std::vector<FeatureIDType> prev_inImg_ids_(0);
    std::vector<int> prev_inImg_lifetime_(0);
    std::vector<cv::Point2f> prev_inImg_points_(0);
    std::vector<cv::Point2f> curr_inImg_points_(0);
    std::vector<cv::Point2f> init_inImg_position_(0);
    std::vector<cv::Mat> prev_imImg_desc_(0);
    //原始特征序列、局内点标示、输出特征序列（将光流没有追踪上的抛出序列）（生成了6个序列）
    removeUnmarkedElements(   
            pts_ids_, track_inliers, prev_inImg_ids_);
    removeUnmarkedElements(
            pts_lifetime_, track_inliers, prev_inImg_lifetime_);
    removeUnmarkedElements(
            prev_pts_, track_inliers, prev_inImg_points_);
    removeUnmarkedElements(
            curr_points, track_inliers, curr_inImg_points_);
    removeUnmarkedElements(
            init_pts_, track_inliers, init_inImg_position_);
    removeUnmarkedElements(
            vOrbDescriptors, track_inliers, prev_imImg_desc_);

    // Number of features left after tracking.
    after_tracking = curr_inImg_points_.size();

    // debug log
    if (0 == after_tracking) {
        printf("No feature is tracked !");
        std::vector<cv::Point2f>().swap(prev_pts_);
        std::vector<cv::Point2f>().swap(curr_pts_);
        std::vector<FeatureIDType>().swap(pts_ids_);
        std::vector<int>().swap(pts_lifetime_);
        std::vector<cv::Point2f>().swap(init_pts_);
        std::vector<cv::Mat>().swap(vOrbDescriptors);
        return;
    }

    // Using reverse LK optical flow tracking to eliminate outliers
    std::vector<unsigned char> reverse_inliers(curr_inImg_points_.size());
    std::vector<cv::Point2f> prev_pts_cpy(prev_inImg_points_);
    calcOpticalFlowPyrLK(
        curr_pyramid_, prev_pyramid_, 
        curr_inImg_points_, prev_pts_cpy,
        reverse_inliers, cv::noArray(),
        win_size,
        pyr_levels,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 15, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    // Mark those tracked points out of the image region
    // as untracked.
    for (int i = 0; i < prev_pts_cpy.size(); ++i) {  
        if (reverse_inliers[i] == 0) continue;
        if (prev_pts_cpy[i].y < 0 ||
            prev_pts_cpy[i].y > prev_pyramid_[0].rows-1 ||
            prev_pts_cpy[i].x < 0 ||
            prev_pts_cpy[i].x > prev_pyramid_[0].cols-1) {
            reverse_inliers[i] = 0;
            continue;
        }
        float dis = cv::norm(prev_pts_cpy[i]-prev_inImg_points_[i]);
        if (dis > 1)    
            reverse_inliers[i] = 0;
    }
    // Remove outliers
    std::vector<FeatureIDType> prev_inImg_ids(0);
    std::vector<int> prev_inImg_lifetime(0);
    std::vector<cv::Point2f> prev_inImg_points(0);
    std::vector<cv::Point2f> curr_inImg_points(0);
    std::vector<cv::Point2f> init_inImg_position(0);
    std::vector<cv::Mat> prev_imImg_desc(0);
    removeUnmarkedElements(   
            prev_inImg_ids_, reverse_inliers, prev_inImg_ids);
    removeUnmarkedElements(
            prev_inImg_lifetime_, reverse_inliers, prev_inImg_lifetime);
    removeUnmarkedElements(
            prev_inImg_points_, reverse_inliers, prev_inImg_points);
    removeUnmarkedElements(
            curr_inImg_points_, reverse_inliers, curr_inImg_points);
    removeUnmarkedElements(
            init_inImg_position_, reverse_inliers, init_inImg_position);
    removeUnmarkedElements(
            prev_imImg_desc_, reverse_inliers, prev_imImg_desc);
    // Number of features left after tracking.
    after_tracking = curr_inImg_points.size();
    // debug log
    if (0 == after_tracking) {
        printf("No feature is tracked !");
        std::vector<cv::Point2f>().swap(prev_pts_);
        std::vector<cv::Point2f>().swap(curr_pts_);
        std::vector<FeatureIDType>().swap(pts_ids_);
        std::vector<int>().swap(pts_lifetime_);
        std::vector<cv::Point2f>().swap(init_pts_);
        std::vector<cv::Mat>().swap(vOrbDescriptors);
        return;
    }

    // Mark as outliers if descriptor distance is too large
    std::vector<int> levels(prev_inImg_points.size(), 0);
    cv::Mat prevDescriptors, currDescriptors;
    //currdecriptorsptc在processimage中已经确定
    //描述符存储到currdescriptor中
    //错误处理
    if (!currORBDescriptor_ptr->computeDescriptors(curr_inImg_points, levels, currDescriptors)) {
        cerr << "error happen while compute descriptors" << endl;
        std::vector<cv::Point2f>().swap(prev_pts_);
        std::vector<cv::Point2f>().swap(curr_pts_);
        std::vector<FeatureIDType>().swap(pts_ids_);
        std::vector<int>().swap(pts_lifetime_);
        std::vector<cv::Point2f>().swap(init_pts_);
        std::vector<cv::Mat>().swap(vOrbDescriptors);
        return;
    }
    std::vector<int> vDis;
    for (int j = 0; j < currDescriptors.rows; ++j) {
        int dis = ORBdescriptor::computeDescriptorDistance(
                prev_imImg_desc[j], currDescriptors.row(j));
        vDis.push_back(dis);
    }
    //通过描述符距离判断是否局内点
    std::vector<unsigned char> desc_inliers(prev_inImg_points.size(), 0);
    for (int i = 0; i < prev_inImg_points.size(); i++) {
        if (vDis[i]<=58)  
            desc_inliers[i] = 1;
    }

    // Remove outliers
    std::vector<FeatureIDType> prev_tracked_ids(0);
    std::vector<int> prev_tracked_lifetime(0);
    std::vector<cv::Point2f> prev_tracked_points(0);
    std::vector<cv::Point2f> curr_tracked_points(0);
    std::vector<cv::Point2f> init_tracked_position(0);
    std::vector<cv::Mat> prev_tracked_desc(0);
    removeUnmarkedElements(    
            prev_inImg_ids, desc_inliers, prev_tracked_ids);
    removeUnmarkedElements(
            prev_inImg_lifetime, desc_inliers, prev_tracked_lifetime);
    removeUnmarkedElements(
            prev_inImg_points, desc_inliers, prev_tracked_points);
    removeUnmarkedElements(
            curr_inImg_points, desc_inliers, curr_tracked_points);
    removeUnmarkedElements(
            init_inImg_position, desc_inliers, init_tracked_position);
    removeUnmarkedElements(
            prev_imImg_desc, desc_inliers, prev_tracked_desc);

    // Return if not enough inliers
    //释放内存
    if ( prev_tracked_points.size()==0 ){
        printf("No feature is tracked after descriptor matching!\n");
        std::vector<cv::Point2f>().swap(prev_pts_);
        std::vector<cv::Point2f>().swap(curr_pts_);
        std::vector<FeatureIDType>().swap(pts_ids_);
        std::vector<int>().swap(pts_lifetime_);
        std::vector<cv::Point2f>().swap(init_pts_);
        std::vector<cv::Mat>().swap(vOrbDescriptors);
        return;
    }

    // Further remove outliers by RANSAC.
    std::vector<cv::Point2f> prev_tracked_unpts(prev_tracked_points.size());
    std::vector<cv::Point2f> curr_tracked_unpts(curr_tracked_points.size());
    undistortPoints(
            prev_tracked_points, cam_intrinsics,
            camera_d_OPENCV.at(0), prev_tracked_unpts, 
            cv::Matx33d::eye(), cam_intrinsics);
    undistortPoints(
            curr_tracked_points, cam_intrinsics,
            camera_d_OPENCV.at(0), curr_tracked_unpts, 
            cv::Matx33d::eye(), cam_intrinsics);

    std::vector<unsigned char> ransac_inliers;

    findFundamentalMat(
            prev_tracked_unpts, curr_tracked_unpts,
            cv::FM_RANSAC, 1.0, 0.99, ransac_inliers);

    // Remove outliers
    std::vector<FeatureIDType> prev_matched_ids(0);
    std::vector<int> prev_matched_lifetime(0);
    std::vector<cv::Point2f> prev_matched_points(0);
    std::vector<cv::Point2f> curr_matched_points(0);
    std::vector<cv::Point2f> init_matched_position(0);
    std::vector<cv::Mat> prev_matched_desc(0);
    removeUnmarkedElements(
            prev_tracked_ids, ransac_inliers, prev_matched_ids);
    removeUnmarkedElements(
            prev_tracked_lifetime, ransac_inliers, prev_matched_lifetime);
    removeUnmarkedElements(
            prev_tracked_points, ransac_inliers, prev_matched_points);
    removeUnmarkedElements(
            curr_tracked_points, ransac_inliers, curr_matched_points);
    removeUnmarkedElements(
            init_tracked_position, ransac_inliers, init_matched_position);
    removeUnmarkedElements(
            prev_tracked_desc, ransac_inliers, prev_matched_desc);

    // Number of matched features left after RANSAC.
    after_ransac = curr_matched_points.size();

    // debug log
    if (0 == after_ransac) {
        printf("No feature survive after RANSAC !");
        std::vector<cv::Point2f>().swap(prev_pts_);
        std::vector<cv::Point2f>().swap(curr_pts_);
        std::vector<FeatureIDType>().swap(pts_ids_);
        std::vector<int>().swap(pts_lifetime_);
        std::vector<cv::Point2f>().swap(init_pts_);
        std::vector<cv::Mat>().swap(vOrbDescriptors);
        return;
    }

    // Puts tracked and mateched points into grids
    std::vector<cv::Point2f>().swap(prev_pts_);
    std::vector<cv::Point2f>().swap(curr_pts_);
    std::vector<FeatureIDType>().swap(pts_ids_);
    std::vector<int>().swap(pts_lifetime_);
    std::vector<cv::Point2f>().swap(init_pts_);
    std::vector<cv::Mat>().swap(vOrbDescriptors);
    for (int i = 0; i < curr_matched_points.size(); ++i) {
        prev_pts_.push_back(prev_matched_points[i]);    
        curr_pts_.push_back(curr_matched_points[i]);
        pts_ids_.push_back(prev_matched_ids[i]);
        pts_lifetime_.push_back(++prev_matched_lifetime[i]);
        init_pts_.push_back(init_matched_position[i]);
        vOrbDescriptors.push_back(prev_matched_desc[i]);
    }

    return;
}

bool TrackLARVIO::initializeFirstFeatures(const std::vector<ImuData>& imu_msg_buffer) {

    //通过先前帧和imu推算当前旋转
    // Integrate gyro data to get a guess of ratation between current and previous image
    integrateImuData(R_Prev2Curr, imu_msg_buffer);

    // Pridict features in current image
    std::vector<cv::Point2f> curr_pts(0);
    predictFeatureTracking(new_pts_, R_Prev2Curr, cam_intrinsics, curr_pts);
    
    // Using LK optical flow to track feaures
    std::vector<unsigned char> track_inliers(new_pts_.size());
    calcOpticalFlowPyrLK(
        prev_pyramid_, curr_pyramid_,
        new_pts_, curr_pts,
        track_inliers, cv::noArray(),
        win_size,
        pyr_levels,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 15, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // Mark those tracked points out of the image region
    // as untracked.
    for (int i = 0; i < curr_pts.size(); ++i) {  
        if (track_inliers[i] == 0) continue;
        if (curr_pts[i].y < 0 ||
            curr_pts[i].y > curr_img_ptr->image.rows-1 ||
            curr_pts[i].x < 0 ||
            curr_pts[i].x > curr_img_ptr->image.cols-1)
            track_inliers[i] = 0;
    }

    // Remove outliers
    std::vector<cv::Point2f> prev_pts_inImg_(0);
    std::vector<cv::Point2f> curr_pts_inImg_(0);
    removeUnmarkedElements(    
            new_pts_, track_inliers, prev_pts_inImg_);
    removeUnmarkedElements(
            curr_pts, track_inliers, curr_pts_inImg_);

    // Return if not enough inliers
    if ( prev_pts_inImg_.size()<20 )
        return false;

    // Using reverse LK optical flow tracking to eliminate outliers
    std::vector<unsigned char> reverse_inliers(curr_pts_inImg_.size());
    std::vector<cv::Point2f> prev_pts_cpy(prev_pts_inImg_);
    calcOpticalFlowPyrLK(
        curr_pyramid_, prev_pyramid_, 
        curr_pts_inImg_, prev_pts_cpy,
        reverse_inliers, cv::noArray(),
        win_size,
        pyr_levels,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 15, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    // Mark those tracked points out of the image region
    // as untracked.
    for (int i = 0; i < prev_pts_cpy.size(); ++i) {  
        if (reverse_inliers[i] == 0) continue;
        if (prev_pts_cpy[i].y < 0 ||
            prev_pts_cpy[i].y > prev_pyramid_[0].rows-1 ||
            prev_pts_cpy[i].x < 0 ||
            prev_pts_cpy[i].x > prev_pyramid_[0].cols-1) {
            reverse_inliers[i] = 0;
            continue;
        }
        float dis = cv::norm(prev_pts_cpy[i]-prev_pts_inImg_[i]);
        if (dis > 1)    
            reverse_inliers[i] = 0;
    }
    // Remove outliers
    std::vector<cv::Point2f> prev_pts_inImg(0);
    std::vector<cv::Point2f> curr_pts_inImg(0);
    removeUnmarkedElements(   
            prev_pts_inImg_, reverse_inliers, prev_pts_inImg);
    removeUnmarkedElements(
            curr_pts_inImg_, reverse_inliers, curr_pts_inImg);
    // Return if not enough inliers
    if ( prev_pts_inImg.size()<20 )
        return false;

    // Mark as outliers if descriptor distance is too large
    std::vector<int> levels(prev_pts_inImg.size(), 0);
    cv::Mat prevDescriptors, currDescriptors;
    if (!prevORBDescriptor_ptr->computeDescriptors(prev_pts_inImg, levels, prevDescriptors) ||
        !currORBDescriptor_ptr->computeDescriptors(curr_pts_inImg, levels, currDescriptors)) {
        cerr << "error happen while compute descriptors" << endl;
        return false;
    }
    std::vector<int> vDis;
    for (int j = 0; j < currDescriptors.rows; ++j) {
        int dis = ORBdescriptor::computeDescriptorDistance(
                prevDescriptors.row(j), currDescriptors.row(j));
        vDis.push_back(dis);
    }
    std::vector<unsigned char> desc_inliers(prev_pts_inImg.size(), 0);
    std::vector<cv::Mat> desc_first(0);
    for (int i = 0; i < prev_pts_inImg.size(); i++) {
        if (vDis[i]<=58) {  
            desc_inliers[i] = 1;
            desc_first.push_back(prevDescriptors.row(i));
        }
    }

    // Remove outliers
    std::vector<cv::Point2f> prev_pts_inlier(0);
    std::vector<cv::Point2f> curr_pts_inlier(0);
    removeUnmarkedElements(   
            prev_pts_inImg, desc_inliers, prev_pts_inlier);
    removeUnmarkedElements(
            curr_pts_inImg, desc_inliers, curr_pts_inlier);

    // Return if not enough inliers
    if ( prev_pts_inlier.size()<20 )
        return false;

    // Undistort inliers
    std::vector<cv::Point2f> prev_unpts_inlier(prev_pts_inlier.size());
    std::vector<cv::Point2f> curr_unpts_inlier(curr_pts_inlier.size());
    undistortPoints(
            prev_pts_inlier, cam_intrinsics,
            camera_d_OPENCV.at(0), prev_unpts_inlier, 
            cv::Matx33d::eye(), cam_intrinsics);
    undistortPoints(
            curr_pts_inlier, cam_intrinsics,
            camera_d_OPENCV.at(0), curr_unpts_inlier, 
            cv::Matx33d::eye(), cam_intrinsics);

    std::vector<unsigned char> ransac_inliers;
    findFundamentalMat(
            prev_unpts_inlier, curr_unpts_inlier,
            cv::FM_RANSAC, 1.0, 0.99, ransac_inliers);

    std::vector<cv::Point2f> prev_pts_matched(0);
    std::vector<cv::Point2f> curr_pts_matched(0);
    std::vector<cv::Mat> prev_desc_matched(0);
    removeUnmarkedElements(
            prev_pts_inlier, ransac_inliers, prev_pts_matched);
    removeUnmarkedElements(
            curr_pts_inlier, ransac_inliers, curr_pts_matched);
    //ransac局外点
    removeUnmarkedElements(
            desc_first, ransac_inliers, prev_desc_matched);

    // Features initialized failed if less than 20 inliers are tracked
    if ( curr_pts_matched.size()<20 )
        return false;
    // and set their ids and lifetime
    std::vector<cv::Point2f>().swap(prev_pts_);
    std::vector<cv::Point2f>().swap(curr_pts_);
    std::vector<FeatureIDType>().swap(pts_ids_);
    std::vector<int>().swap(pts_lifetime_);
    std::vector<cv::Point2f>().swap(init_pts_);
    std::vector<cv::Mat>().swap(vOrbDescriptors);
    for (int i = 0; i < prev_pts_matched.size(); ++i) {
        prev_pts_.push_back(prev_pts_matched[i]);
        init_pts_.push_back(cv::Point2f(-1,-1));       
        curr_pts_.push_back(curr_pts_matched[i]);
        pts_ids_.push_back(next_feature_id++);
        pts_lifetime_.push_back(2);
        vOrbDescriptors.push_back(prev_desc_matched[i]);
    }

    // Clear new_pts_
    std::vector<cv::Point2f>().swap(new_pts_);

    return true;
}

bool TrackLARVIO::initializeFirstFrame() {

    /*
    cv::Matx33d cameK=camera_k_OPENCV.at(0);
    cam_intrinsics(0)=cameK(0,0);
    cam_intrinsics(1)=cameK(1,1);
    cam_intrinsics(2)=cameK(0,2);
    cam_intrinsics(3)=cameK(1,2);
    */

    // Get current image
    //std::cout <<  img_pyramid_last.size()<< std::endl;
    const cv::Mat& img = curr_pyramid_[0];
    // Detect new features on the frist image.
    std::vector<cv::Point2f>().swap(new_pts_);
     std::cout << "ssssssssssssssssssss" << std::endl;
    cv::goodFeaturesToTrack(img, new_pts_, num_features, 0.01, min_px_dist);
     std::cout << "ssssssssssssssssssss" << std::endl;
    /*
    std::vector<cv::KeyPoint> pts0_ext;
    //Grider_FAST::perform_griding(img, pts0_ext, num_features, grid_x, grid_y, threshold, true);
    
    std::vector<cv::KeyPoint> kpts0_new;
    std::vector<cv::Point2f> pts0_new;
    for(auto& kpt : pts0_ext) {
        // See if there is a point at this location
        if(grid_2d((int)(kpt.pt.y/min_px_dist),(int)(kpt.pt.x/min_px_dist)) == 1)
            continue;
        // Else lets add it!
        kpts0_new.push_back(kpt);
        pts0_new.push_back(kpt.pt);
        grid_2d((int)(kpt.pt.y/min_px_dist),(int)(kpt.pt.x/min_px_dist)) = 1;
    }

    new_pts_.assign(pts0_new.begin(),pts0_new.end());
    */
    // Initialize last publish time
    last_pub_time = curr_img_ptr->timeStampToSec;
    
    if (new_pts_.size()>20)
        return true;
    else
        return false;
}

void TrackLARVIO::predictFeatureTracking(
    const std::vector<cv::Point2f>& input_pts,
    const cv::Matx33f& R_p_c,
    const cv::Vec4d& intrinsics,
    std::vector<cv::Point2f>& compensated_pts) {
    // Return directly if there are no input features.
    if (input_pts.size() == 0) {
        compensated_pts.clear();
        return;
    }
    compensated_pts.resize(input_pts.size());

    // Intrinsic matrix(相机内参).
    cv::Matx33f K(
        intrinsics[0], 0.0, intrinsics[2],
        0.0, intrinsics[1], intrinsics[3],
        0.0, 0.0, 1.0);
    //将上一帧中的特这点坐标通过相机内参和相机旋转矩阵映射到当前帧中（像素坐标）
    cv::Matx33f H = K * R_p_c * K.inv();  

    for (int i = 0; i < input_pts.size(); ++i) {
        //补成齐次坐标
        cv::Vec3f p1(input_pts[i].x, input_pts[i].y, 1.0f);
        cv::Vec3f p2 = H * p1;
        compensated_pts[i].x = p2[0] / p2[2];
        compensated_pts[i].y = p2[1] / p2[2];
    }

    return;
}

void TrackLARVIO::createImagePyramids() {
    const cv::Mat& curr_img = curr_img_ptr->image;
    // CLAHE
    cv::Mat img_;
    img_ = curr_img;

    // Get Pyramid
    cv::buildOpticalFlowPyramid(img_, curr_pyramid_, win_size, pyr_levels);   
}

void TrackLARVIO::undistortPoints(
    const std::vector<cv::Point2f>& pts_in,
    const cv::Vec4d& intrinsics,
    const cv::Vec4d& distortion_coeffs,
    std::vector<cv::Point2f>& pts_out,
    const cv::Matx33d &rectification_matrix,
    const cv::Vec4d &new_intrinsics) {
    if (pts_in.size() == 0) return;

    const cv::Matx33d K(
            intrinsics[0], 0.0, intrinsics[2],
            0.0, intrinsics[1], intrinsics[3],
            0.0, 0.0, 1.0);

    const cv::Matx33d K_new(
            new_intrinsics[0], 0.0, new_intrinsics[2],
            0.0, new_intrinsics[1], new_intrinsics[3],
            0.0, 0.0, 1.0);
    
    if (this->camera_fisheye.at(0)) {
        cv::fisheye::undistortPoints(pts_in, pts_out, K, distortion_coeffs,rectification_matrix, K_new);\
        return;       
        }
    cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,rectification_matrix, K_new);
    return;    
        
}

//imu积分
void TrackLARVIO::integrateImuData(cv::Matx33f& cam_R_p2c,const std::vector<ImuData>& imu_msg_buffer) {
    // Find the start and the end limit within the imu msg buffer.
    auto begin_iter = imu_msg_buffer.begin();
    //begin在上一帧前0.0049s以内
    while (begin_iter != imu_msg_buffer.end()) {
    if (begin_iter->timeStampToSec-
            prev_img_ptr->timeStampToSec < -0.0049)
        ++begin_iter;
    else
        break;
    }

    auto end_iter = begin_iter;
    //end在当前帧后0.0049s以内
    while (end_iter != imu_msg_buffer.end()) {
    if (end_iter->timeStampToSec-
            curr_img_ptr->timeStampToSec < 0.0049)
        ++end_iter;
    else
        break;
    }
    //计算角速度均值
    // Compute the mean angular velocity in the IMU frame.
    cv::Vec3f mean_ang_vel(0.0, 0.0, 0.0);
    for (auto iter = begin_iter; iter < end_iter; ++iter)
    mean_ang_vel += cv::Vec3f(iter->angular_velocity[0],
        iter->angular_velocity[1], iter->angular_velocity[2]);

    if (end_iter-begin_iter > 0)
    mean_ang_vel *= 1.0f / (end_iter-begin_iter);

    // Transform the mean angular velocity from the IMU
    // frame to the cam0 and cam1 frames.
    //imu坐标系转换到相机坐标系
    cv::Vec3f cam_mean_ang_vel = R_cam_imu.t() * mean_ang_vel;

    // Compute the relative rotation.
    double dtime = curr_img_ptr->timeStampToSec-
        prev_img_ptr->timeStampToSec;
    cv::Rodrigues(cam_mean_ang_vel*dtime, cam_R_p2c);
    cam_R_p2c = cam_R_p2c.t();
    //cam_R_p2c是输出的imu计算的相机旋转矩阵
    return;
}

//原始
void TrackLARVIO::feed_monocular(double timestamp, cv::Mat &img, size_t cam_id) {

    // Start timing
    rT1 =  boost::posix_time::microsec_clock::local_time();

    if (!bFirstImg) {
        if ((imu_msg_buffer.begin() != imu_msg_buffer.end()) && 
            (imu_msg_buffer.begin()->timeStampToSec-timestamp <= 0.0)) {
            bFirstImg = true;
            printf("Images from now on will be utilized...\n\n");
        }
        else
            return ;
    }

    ImageDataPtr msgPtr(new ImgData);
    msgPtr->timeStampToSec = timestamp;
    msgPtr->image = img.clone();
    curr_img_ptr=msgPtr;
    curr_img_time=curr_img_ptr->timeStampToSec;
    std::vector<cv::Point2f> good_left;
    std::vector<size_t> good_ids_left;

    // Lock this data feed for this camera
    std::unique_lock<std::mutex> lck(mtx_feeds.at(cam_id));

    //直方图均衡化
    // Histogram equalize
    cv::equalizeHist(img, img);

    // Extract the new image pyramid
    createImagePyramids();
    currORBDescriptor_ptr.reset(new ORBdescriptor(curr_pyramid_[0], 2, pyr_levels));
    std::vector<cv::Mat> imgpyr(curr_pyramid_);
    bool haveFeatures = false;
    
    //cv::buildOpticalFlowPyramid(img, imgpyr, win_size, pyr_levels);
    rT2 =  boost::posix_time::microsec_clock::local_time();
   
    if ( FIRST_IMAGE==image_state ) {
        if (initializeFirstFrame())
            image_state = SECOND_IMAGE;
    } else if ( SECOND_IMAGE==image_state ) {
        if ( !initializeFirstFeatures(imu_msg_buffer) ) {
            image_state = FIRST_IMAGE;
        } else {
            // frequency control
            if ( curr_img_time-last_pub_time >= 0.9*(1.0/10) ) {
                // Find new features to be tracked
                findNewFeaturesToBeTracked();

                // Det processed feature

                // Publishing msgs

                haveFeatures = true;
            }

            image_state = OTHER_IMAGES;
        }
    } else if ( OTHER_IMAGES==image_state ) {
        // Integrate gyro data to get a guess of rotation between current and previous image
        integrateImuData(R_Prev2Curr, imu_msg_buffer);

        // Tracking features
        trackFeatures();

        // Track new features extracted in last image, and add them into the gird
        trackNewFeatures();

        // frequency control
        if ( curr_img_time-last_pub_time >= 0.9*(1.0/10) ) {
            // Find new features to be tracked
            findNewFeaturesToBeTracked();

            // Det processed feature

            // Loop through all left points

            for(size_t i=0; i<prev_pts_.size(); i++) {
                good_left.push_back(prev_pts_[i]);
                good_ids_left.push_back(pts_ids_[i]);
            }

            //===================================================================================
            //===================================================================================


            // Update our feature database, with theses new observations
            for(size_t i=0; i<good_left.size(); i++) {
                cv::Point2f npt_l = undistort_point(good_left.at(i), cam_id);
                database->update_feature(good_ids_left.at(i), timestamp, cam_id,
                                        good_left.at(i).x, good_left.at(i).y,
                                        npt_l.x, npt_l.y);
            }


            haveFeatures = true;
        }
    }
     
    rT3 =  boost::posix_time::microsec_clock::local_time();

    rT4 =  boost::posix_time::microsec_clock::local_time();

    //===================================================================================
    //===================================================================================

    // If any of our mask is empty, that means we didn't have enough to do ransac, so just return

    // Get our "good tracks"
    /*
    std::vector<cv::Point2f> good_left;
    std::vector<size_t> good_ids_left;

    // Loop through all left points

    for(size_t i=0; i<prev_pts_.size(); i++) {
        good_left.push_back(prev_pts_[i]);
        good_ids_left.push_back(pts_ids_[i]);
    }

    //===================================================================================
    //===================================================================================


    // Update our feature database, with theses new observations
    for(size_t i=0; i<good_left.size(); i++) {
        cv::Point2f npt_l = undistort_point(good_left.at(i), cam_id);
        database->update_feature(good_ids_left.at(i), timestamp, cam_id,
                                 good_left.at(i).x, good_left.at(i).y,
                                 npt_l.x, npt_l.y);
    }
    */
    // Update the previous image and previous features.
    prev_img_ptr = curr_img_ptr;
    std::swap(prev_pyramid_,curr_pyramid_);
    prevORBDescriptor_ptr = currORBDescriptor_ptr;

    // Initialize the current features to empty vectors.
    std::swap(prev_pts_,curr_pts_);
    std::vector<cv::Point2f>().swap(curr_pts_);

    prev_img_time = curr_img_time;

    // Move forward in time
    img_last[cam_id] = img.clone();
    img_pyramid_last[cam_id] = imgpyr;
    ids_last[cam_id] = good_ids_left;

    std::vector<cv::Point2f>().swap(good_left);
    std::vector<size_t>().swap(good_ids_left);
    
    rT5 =  boost::posix_time::microsec_clock::local_time();

}

void TrackLARVIO::setcameraintrinsics(std::map<size_t,Eigen::VectorXd> camera_calib, const Eigen::Matrix<double, 4, 1> camex) {

    cam_intrinsics(0)=camera_calib.at(0)[0];
    cam_intrinsics(1)=camera_calib.at(0)[1];
    cam_intrinsics(2)=camera_calib.at(0)[2];
    cam_intrinsics(3)=camera_calib.at(0)[3];
    
    Eigen::Matrix<double, 3, 1> temp=camex.block(0,0,3,1);
    Eigen::Matrix<double, 3, 3> q_x;
            q_x << 0, -temp(2), temp(1),
                temp(2), 0, -temp(0),
                -temp(1), temp(0), 0;

    Eigen::MatrixXd temp1 = (2 * std::pow(camex(3, 0), 2) - 1) * Eigen::MatrixXd::Identity(3, 3)
                              - 2 * camex(3, 0) * q_x +
                              2 * camex.block(0, 0, 3, 1) * (camex.block(0, 0, 3, 1).transpose());
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){

            R_cam_imu(i,j)=temp1(i,j);

        }
    }

}

void TrackLARVIO::feedimu(double timestamp, Eigen::Vector3d wm, Eigen::Vector3d am){

    imu_msg_buffer.push_back(ImuData(timestamp, wm[0], wm[1], wm[2], am[0], am[1], am[2]));

}

void TrackLARVIO::feed_stereo(double timestamp, cv::Mat &img_left, cv::Mat &img_right, size_t cam_id_left, size_t cam_id_right){

}

void TrackLARVIO::perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, std::vector<cv::KeyPoint> &pts0, std::vector<size_t> &ids0){

}

void TrackLARVIO::perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &pts0,
                                      std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1){

                                      }

void TrackLARVIO::perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &pts0,
                              std::vector<cv::KeyPoint> &pts1, size_t id0, size_t id1, std::vector<uchar> &mask_out){

                              }