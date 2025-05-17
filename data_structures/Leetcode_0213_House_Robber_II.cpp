#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    // Modify the dfs function to take a const reference to avoid temporary object binding issues
    int dfs(const vector<int>& number) {
        int n = number.size();
        if (n == 0) return 0;  // No houses to rob
        if (n == 1) return number[0];  // Only one house to rob
        
        vector<int> dp(n, 0);  // DP array to store maximum profit at each step
        dp[0] = number[0];  // Base case: First house can only be robbed for its own value
        dp[1] = max(number[0], number[1]);  // The second house is the maximum of the first and second house
        
        for (int i = 2; i < n; ++i) {
            dp[i] = max(dp[i - 2] + number[i], dp[i - 1]);  // Either rob current house or skip it
        }
        
        return dp[n - 1];  // The last element contains the maximum profit
    }

    int rob(vector<int>& nums) {
        int n = nums.size();
        if (n == 1) return nums[0];  // If there's only one house, return the value of that house
        if (n == 0) return 0;  // No houses to rob
        
        // The result is the maximum of three possibilities:
        // 1. Rob from the second house to the last house (i.e., exclude the first house)
        // 2. Rob from the first house to the second-to-last house (i.e., exclude the last house)
        // 3. Just rob the first house (if there's only one house)
        return max(nums[0], max(dfs(vector<int>(nums.begin() + 1, nums.end())), dfs(vector<int>(nums.begin(), nums.end() - 1))));
    }
};

int main() {
    Solution sol;
    vector<int> nums = {1, 2, 3, 1};
    int result = sol.rob(nums);
    cout << result << endl;  // Expected output: 4
    
    vector<int> nums2 = {0, 0};
    int result2 = sol.rob(nums2);
    cout << result2 << endl;  // Expected output: 0
    
    return 0;
}


// g++ -std=c++17 Leetcode_0213_House_Robber_II.cpp -o test