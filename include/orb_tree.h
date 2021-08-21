#ifndef _ORB_TREE_H
#define _ORB_TREE_H
#include "comment_include.h"

namespace ZOULEE {
    class Nodes_tree{
        public:
            Nodes_tree();
            Nodes_tree(std::vector<cv::KeyPoint> &keypoints_, int max_x, int max_y);
            cv::Point2i get_min();
            cv::Point2i get_max();
            bool is_able_to_devide();
            std::list<cv::KeyPoint*> keypoints;
            int get_max_response_index();
            void set_max(int x, int y);
            void set_min(int x, int y);
            void set_max(cv::Point2i max_);
            void set_min(cv::Point2i max_);

            void set_able_to_devide(bool able);
            void set_max_response_index(int index);
            // float max_response;

        private:
            cv::Point2i min;
            cv::Point2i max;
            bool devide_able;
            int max_key_index;

    };
    // void drawkey_list_t(std::list<cv::KeyPoint*> keypoints);
    void make_tree(cv::Mat img, std::vector<cv::KeyPoint> &keypoints, u16 keypoints_num);
    void one_t_four(Nodes_tree node, std::list<Nodes_tree> &nodes);
    // void drawkey_list(std::list<cv::KeyPoint> keypoints);
    // void drawkey_vector(std::vector<cv::KeyPoint> keypoints);
}

#endif // !_ORB_TREE_H
