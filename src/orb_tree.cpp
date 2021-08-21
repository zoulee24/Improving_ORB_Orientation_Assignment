#include "orb_tree.h"

using namespace std;
using namespace cv;

namespace ZOULEE {

    void Nodes_tree::set_able_to_devide(bool able){
        devide_able = able;
    }

    void Nodes_tree::set_max_response_index(int index){
        max_key_index = index;
    }

    Nodes_tree::Nodes_tree(){
        devide_able = false;
        max_key_index = -1;
    }

    void Nodes_tree::set_max(int x, int y){
        max.x = x;
        max.y = y;
    }
    void Nodes_tree::set_max(Point2i max_){
        max = max_;
    }

    void Nodes_tree::set_min(int x, int y){
        min.x = x;
        min.y = y;
    }
    void Nodes_tree::set_min(Point2i min_){
        min = min_;
    }

    Point2i Nodes_tree::get_min(){
        return min;
    }

    Point2i Nodes_tree::get_max(){
        return max;
    }

    int Nodes_tree::get_max_response_index(){
        return max_key_index;
    }

    bool Nodes_tree::is_able_to_devide(){
        return devide_able;
    }

    Nodes_tree::Nodes_tree(vector<KeyPoint> &keypoints_, int max_x, int max_y){
        for(auto &key : keypoints_){
            keypoints.push_back(&key);
        }
        
        min.x = 0;
        min.y = 0;
        max.x = max_x;
        max.y = max_y;
        if(keypoints.empty()){
            devide_able = false;
        }
        else{
            devide_able = true;
        }
        max_key_index = -1;
    }

    void make_tree(Mat img, vector<KeyPoint> &keypoints, u16 keypoints_num){
        if(keypoints.size() < keypoints_num){
            // cout << "orb_tree.cpp(make_tree)  keypoints_num too large" << endl << endl;
            return;
        }
        Nodes_tree all(keypoints, img.cols, img.rows);
        list<Nodes_tree> nodes;
        nodes.push_back(all);
        while (nodes.size() < keypoints_num)
        {
            one_t_four(nodes.back(), nodes);
        }

        std::list<cv::KeyPoint*>::iterator it;
        const int N = nodes.size();
        keypoints.resize(N);
        int count = 0;
        
        for(auto node : nodes){
            it = node.keypoints.begin();
            if (node.get_max_response_index() > 0){
                advance(it, node.get_max_response_index() );
            }
            keypoints[count++] = **it;
        }
        // user_main_debug("orb_tree level %d Keypoint size = %d", keypoints[0].octave, keypoints.size());
    }

    void one_t_four(Nodes_tree node, list<Nodes_tree> &nodes){
        if(node.is_able_to_devide())
        {
            Nodes_tree n1, n2, n3, n4;
            const u16 half_x = (node.get_max().x + node.get_min().x) / 2;
            const u16 half_y = (node.get_max().y + node.get_min().y) / 2;
            n1.set_min(node.get_min());
            n1.set_max(half_x, half_y);
            n2.set_min(half_x, node.get_min().y);
            n2.set_max(node.get_max().x, half_y);
            n3.set_min(node.get_min().x, half_y);
            n3.set_max(half_x, node.get_max().y);
            n4.set_min(half_x, half_y);
            n4.set_max(node.get_max());

            float n1_max_response = 0.0f, n2_max_response = 0.0f, n3_max_response = 0.0f, n4_max_response = 0.0f;
            float n1_max_response_last = 0.0f, n2_max_response_last = 0.0f, n3_max_response_last = 0.0f, n4_max_response_last = 0.0f;
            int n1_index = -1, n2_index = -1, n3_index = -1, n4_index = -1;
            int N = node.keypoints.size();
            for(list<KeyPoint *>::iterator it = node.keypoints.begin(); it != node.keypoints.end(); it++){
                if((*it)->pt.x < half_x){
                    if((*it)->pt.y < half_y){
                        n1_index++;
                        n1_max_response_last = max((*it)->response, n1_max_response);
                        if(n1_max_response < n1_max_response_last){
                            n1.set_max_response_index(n1_index);
                            n1_max_response = n1_max_response_last;
                            // n1.max_response = n1_max_response_last;
                        }
                        n1.keypoints.push_back(*it);
                    }
                    else{
                        n3_index++;
                        n3_max_response_last = max((*it)->response, n3_max_response);
                        if(n3_max_response < n3_max_response_last){
                            n3.set_max_response_index(n3_index);
                            n3_max_response = n3_max_response_last;
                            // n3.max_response = n3_max_response_last;
                        }
                        n3.keypoints.push_back(*it);
                    }
                }
                else{
                    if((*it)->pt.y < half_y){
                        n2_index++;
                        n2_max_response_last = max((*it)->response, n2_max_response);
                        if(n2_max_response < n2_max_response_last){
                            n2.set_max_response_index(n2_index);
                            n2_max_response = n2_max_response_last;
                            // n2.max_response = n2_max_response_last;
                        }
                        n2.keypoints.push_back(*it);
                    }
                    else{
                        n4_index++;
                        n4_max_response_last = max((*it)->response, n4_max_response);
                        if(n4_max_response < n4_max_response_last){
                            n4.set_max_response_index(n4_index);
                            n4_max_response = n4_max_response_last;
                            // n4.max_response = n4_max_response_last;
                        }
                        n4.keypoints.push_back(*it);
                    }
                }
            }
            if(!n1.keypoints.empty()){
                if(n1.keypoints.size()>1){
                    n1.set_able_to_devide(true);
                }
                // drawkey_list_t(n1.keypoints);
                nodes.push_front(n1);
            }
            if(!n2.keypoints.empty()){
                if(n2.keypoints.size()>1){
                    n2.set_able_to_devide(true);
                }
                // drawkey_list_t(n2.keypoints);
                nodes.push_front(n2);
            }
            if(!n3.keypoints.empty()){
                if(n3.keypoints.size()>1){
                    n3.set_able_to_devide(true);
                }
                // drawkey_list_t(n3.keypoints);
                nodes.push_front(n3);
            }
            if(!n4.keypoints.empty()){
                if(n4.keypoints.size()>1){
                    n4.set_able_to_devide(true);
                }
                // drawkey_list_t(n4.keypoints);
                nodes.push_front(n4);
            }
            nodes.pop_back();
        }
        else{
            nodes.push_front(node);
            nodes.pop_back();
        }
    }

}