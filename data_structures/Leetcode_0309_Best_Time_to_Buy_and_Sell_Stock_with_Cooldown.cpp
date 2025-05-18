#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        if (n < 2) return 0;

        // dp[i][0] = hold
        // dp[i][1] = sold (cooldown)
        // dp[i][2] = rest (not in cooldown)
        vector<vector<int>> dp(n, vector<int>(3, 0));
        
        // Initialize Day 0
        dp[0][0] = -prices[0]; // buy
        dp[0][1] = 0;          // sold (not applicable on day 0)
        dp[0][2] = 0;          // rest

        for (int i = 1; i < n; ++i) {
            dp[i][0] = max(dp[i-1][0], dp[i-1][2] - prices[i]);
            dp[i][1] = dp[i-1][0] + prices[i]; // sell
            dp[i][2] = max(dp[i-1][1], dp[i-1][2]);
        }

        // Max profit is max of last day not holding stock
        return max(dp[n-1][1], dp[n-1][2]);
    }
};

// Test Case
int main() {
    Solution sol;
    vector<int> prices = {1, 2, 3, 0, 2};
    cout << sol.maxProfit(prices) << endl; // Output: 3
    return 0;
}

// g++ -std=c++17 Leetcode_0309_Best_Time_to_Buy_and_Sell_Stock_with_Cooldown.cpp -o test  