#include <iostream>
#include <vector>
#include <algorithm> // For std::max

using namespace std;

class Solution {
public:
    bool canJump(vector<int>& nums) {
        int n = nums.size();
        int cover = 0;
        for (int i = 0; i < n; ++i) {
            if (i > cover) {
                return false;
            }
            int jump = i + nums[i];
            cover = max(cover, jump);
            if (cover >= n - 1) {
                return true;
            }
        }
        return false;
    }
};

int main() {
    Solution sol;
    vector<int> nums = {2, 3, 1, 1, 4};
    bool res = sol.canJump(nums);
    cout << boolalpha << res << endl; // Outputs "true"
    return 0;
}

// g++ -std=c++17 Leetcode_0055_Jump_Game.cpp -o test