#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        long long total_sum = 0;
        for (int num : nums) {
            total_sum += num;
        }

        // Edge case: no way to reach target
        if (abs(target) > total_sum || (target + total_sum) % 2 != 0) {
            return 0;
        }

        long long target_sum = (target + total_sum) / 2;

        // Handle overflow or negative target_sum
        if (target_sum < 0) return 0;

        int n = nums.size();
        vector<vector<long long>> dp(n + 1, vector<long long>(target_sum + 1, 0));

        // Base case: one way to make sum 0 (choose nothing)
        for (int i = 0; i <= n; ++i) {
            dp[i][0] = 1;
        }

        // Fill DP table
        for (int i = 1; i <= n; ++i) {
            for (int j = 0; j <= target_sum; ++j) {
                long long remain = j - nums[i - 1];
                if (remain < 0) {
                    dp[i][j] = dp[i - 1][j];  // Can't include current number
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][remain];  // Include or exclude
                }
            }
        }

        return dp[n][target_sum];
    }
};

// Test Case
int main() {
    Solution sol;
    vector<int> nums = {1, 1, 1, 1, 1};
    int target = 3;
    cout << sol.findTargetSumWays(nums, target) << endl; // Output: 5
    return 0;
}

// g++ -std=c++17 Leetcode_0494_Target_Sum.cpp -o test     