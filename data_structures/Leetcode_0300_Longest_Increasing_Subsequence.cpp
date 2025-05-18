#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) return 0;

        vector<int> LIS(n, 1);  // Initialize LIS array with 1s

        // Iterate through the array from the second-to-last element to the first
        for (int i = n - 2; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                if (nums[i] < nums[j]) {
                    LIS[i] = max(LIS[i], 1 + LIS[j]);  // Update LIS for nums[i]
                }
            }
        }

        // Return the maximum value in the LIS array, which is the length of the longest subsequence
        return *max_element(LIS.begin(), LIS.end());
    }
};

int main() {
    Solution sol;
    vector<int> nums = {10, 9, 2, 5, 3, 7, 101, 18};
    int result = sol.lengthOfLIS(nums);
    cout << result << endl;  // Expected output: 4 (LIS: [2, 3, 7, 101])

    return 0;
}


// g++ -std=c++17 Leetcode_0300_Longest_Increasing_Subsequence.cpp -o test