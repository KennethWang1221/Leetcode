#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) return 0;  // No houses to rob
        if (n == 1) return nums[0];  // Only one house to rob
        
        vector<int> dp(n + 1, 0);  // DP array to store maximum profit at each step
        dp[1] = nums[0];  // Base case: The first house can only be robbed for its own value
        
        // Calculate the maximum profit for each house
        for (int i = 2; i <= n; ++i) {
            dp[i] = max(dp[i - 1], nums[i - 1] + dp[i - 2]);  // Rob the current house or skip it
        }

        return dp[n];  // The last element contains the maximum profit
    }
};

int main() {
    Solution sol;
    vector<int> nums = {1, 2, 3, 1};  // Example input
    int result = sol.rob(nums);
    cout << result << endl;  // Expected output: 4
    
    return 0;
}


// g++ -std=c++17 Leetcode_0198_House_Robber.cpp -o test  