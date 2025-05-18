#include <iostream>
#include <vector>
#include <numeric>

using namespace std;

class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int total_sum = accumulate(nums.begin(), nums.end(), 0);  // Calculate the total sum of the array
        
        // If the total sum is odd, it's impossible to partition into two equal subsets
        if (total_sum % 2 != 0) {
            return false;
        }

        int n = nums.size();
        int target_sum = total_sum / 2;

        // Create a 2D DP table, dp[i][j] will be true if it's possible to achieve sum j using the first i elements
        vector<vector<bool>> dp(n + 1, vector<bool>(target_sum + 1, false));

        // Initialize the first column to true (sum 0 is always possible with an empty subset)
        for (int i = 0; i <= n; ++i) {
            dp[i][0] = true;
        }

        // Fill the DP table
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= target_sum; ++j) {
                int remain = j - nums[i - 1];
                if (remain < 0) {
                    dp[i][j] = dp[i - 1][j];  // Current number is too large, cannot use it
                } else {
                    dp[i][j] = dp[i - 1][j] || dp[i - 1][remain];  // Either don't use or use the current number
                }
            }
        }

        return dp[n][target_sum];  // The answer is in dp[n][target_sum]
    }
};

int main() {
    Solution sol;
    
    vector<int> nums1 = {1, 5, 11, 5};
    cout << (sol.canPartition(nums1) ? "True" : "False") << endl;  // Expected output: True
    
    vector<int> nums2 = {2, 2, 1, 1};
    cout << (sol.canPartition(nums2) ? "True" : "False") << endl;  // Expected output: True

    return 0;
}


// g++ -std=c++17 Leetcode_0416_Partition_Equal_Subset_Sum.cpp -o test