#include <iostream>
#include <vector>
#include <climits>  // For INT_MAX
using namespace std;

class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int n = nums.size();
        int l = 0, total = 0, res = INT_MAX;
        
        for (int r = 0; r < n; r++) {
            total += nums[r];

            while (total >= target) {
                res = min(res, r - l + 1);  // Update the result with the smallest window size
                total -= nums[l];  // Subtract the element at left pointer
                l++;  // Move the left pointer to shrink the window
            }
        }

        return res == INT_MAX ? 0 : res;  // Return 0 if no valid subarray is found
    }
};

int main() {
    Solution solution;
    int target = 7;
    vector<int> nums = {2, 3, 1, 2, 4, 3};

    int result = solution.minSubArrayLen(target, nums);
    cout << "Minimum subarray length: " << result << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_0209_Minimum_Size_Subarray_Sum.cpp -o test