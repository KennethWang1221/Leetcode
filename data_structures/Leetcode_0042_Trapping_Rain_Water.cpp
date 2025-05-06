#include <iostream>
#include <vector>
#include <algorithm>  // for max
using namespace std;

class Solution {
public:
    int trap(vector<int>& height) {
        if (height.empty()) {
            return 0;
        }

        int l = 0, r = height.size() - 1;
        int leftMax = height[l], rightMax = height[r];
        int res = 0;

        while (l < r) {
            if (leftMax <= rightMax) {
                l++;
                leftMax = max(leftMax, height[l]);
                int cur = height[l];
                res += leftMax - cur;
            } else {
                r--;
                rightMax = max(rightMax, height[r]);
                int cur = height[r];
                res += rightMax - cur;
            }
        }

        return res;
    }
};

int main() {
    Solution solution;
    vector<int> height = {0,1,0,2,1,0,1,3,2,1,2,1};

    int result = solution.trap(height);
    cout << "Amount of trapped rain water: " << result << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_0042_Trapping_Rain_Water.cpp -o test