#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    int maxCoins(vector<int>& nums) {
        int n = nums.size();
        vector<int> arr(n + 2); // Add 1s on both ends
        arr[0] = 1;
        arr[n + 1] = 1;
        for (int i = 0; i < n; ++i)
            arr[i + 1] = nums[i];

        vector<vector<int>> dp(n + 2, vector<int>(n + 2, 0));

        // Length l goes from 1 to n
        for (int l = 1; l <= n; ++l) {
            for (int i = 1; i <= n - l + 1; ++i) {
                int j = i + l - 1;
                for (int k = i; k <= j; ++k) {
                    dp[i][j] = max(dp[i][j], dp[i][k - 1] + arr[i - 1] * arr[k] * arr[j + 1] + dp[k + 1][j]);
                }
            }
        }

        return dp[1][n];
    }
};

// Test Case
int main() {
    Solution sol;
    vector<int> nums = {3,1,5,8};
    cout << "Max Coins: " << sol.maxCoins(nums) << endl; // Output: 167
    return 0;
}

// g++ -std=c++17 Leetcode_0312_Burst_Ballons.cpp -o test  