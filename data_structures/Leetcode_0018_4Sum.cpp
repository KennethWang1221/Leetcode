#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> res;
        int n = nums.size();
        if (n < 4) return res;
        
        sort(nums.begin(), nums.end());
        
        for (int i = 0; i < n - 3; ++i) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            
            for (int j = i + 1; j < n - 2; ++j) {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;
                
                int l = j + 1, r = n - 1;
                while (l < r) {
                    long long sum = (long long)nums[i] + nums[j] + nums[l] + nums[r];
                    if (sum < target) {
                        ++l;
                    } else if (sum > target) {
                        --r;
                    } else {
                        res.push_back({nums[i], nums[j], nums[l], nums[r]});
                        ++l;
                        while (l < r && nums[l] == nums[l - 1]) ++l;
                    }
                }
            }
        }
        return res;
    }
};

int main() {
    Solution sol;
    vector<int> nums = {1, 0, -1, 0, -2, 2};
    int target = 0;
    vector<vector<int>> res = sol.fourSum(nums, target);
    
    cout << "Result: ";
    for (auto& quad : res) {
        cout << "[";
        for (size_t i = 0; i < quad.size(); ++i) {
            cout << quad[i];
            if (i != quad.size() - 1) cout << ", ";
        }
        cout << "] ";
    }
    cout << endl; // Expected output: [-2, -1, 1, 2] [-2, 0, 0, 2] [-1, 0, 0, 1]
    
    return 0;
}

// g++ -std=c++17 Leetcode_0018_4Sum.cpp -o test