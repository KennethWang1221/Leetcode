#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        int n = nums.size();
        if (n < 3) return res;
        
        sort(nums.begin(), nums.end());
        
        for (int i = 0; i < n; ++i) {
            if (i > 0 && nums[i] == nums[i-1]) {
                continue; // Skip duplicate for the first element
            }
            int l = i + 1, r = n - 1;
            while (l < r) {
                int sum = nums[i] + nums[l] + nums[r];
                if (sum > 0) {
                    --r;
                } else if (sum < 0) {
                    ++l;
                } else {
                    res.push_back({nums[i], nums[l], nums[r]});
                    ++l;
                    // Skip duplicate for the second element
                    while (l < r && nums[l] == nums[l-1]) {
                        ++l;
                    }
                }
            }
        }
        return res;
    }
};

int main() {
    Solution sol;
    vector<int> nums1 = {-1, 0, 1, 2, -1, -4};
    vector<vector<int>> res1 = sol.threeSum(nums1);
    cout << "Test case 1: ";
    for (auto& triplet : res1) {
        cout << "[";
        for (size_t i = 0; i < triplet.size(); ++i) {
            cout << triplet[i];
            if (i != triplet.size() - 1) cout << ",";
        }
        cout << "] ";
    }
    cout << endl; // Expected output: [-1,-1,2] [-1,0,1]
    
    vector<int> nums2 = {0, 0, 0, 0};
    vector<vector<int>> res2 = sol.threeSum(nums2);
    cout << "Test case 2: ";
    for (auto& triplet : res2) {
        cout << "[";
        for (size_t i = 0; i < triplet.size(); ++i) {
            cout << triplet[i];
            if (i != triplet.size() - 1) cout << ",";
        }
        cout << "] ";
    }
    cout << endl; // Expected output: [0,0,0]
    
    return 0;
}

// g++ -std=c++17 Leetcode_0015_3Sum.cpp -o test