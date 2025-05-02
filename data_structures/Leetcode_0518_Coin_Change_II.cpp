#include <vector>
#include <iostream>

using namespace std;

class Solution {
public:
    int change(int amount, vector<int>& coins) {
        int n = coins.size();
        vector<vector<unsigned long long>> dp(n + 1, vector<unsigned long long>(amount + 1, 0));
        
        // Base case: 1 way to make amount 0 (use no coins)
        for (int i = 0; i <= n; ++i) {
            dp[i][0] = 1;
        }
        
        for (int i = 1; i <= n; ++i) {
            for (int a = 1; a <= amount; ++a) {
                int remain = a - coins[i - 1];
                if (remain < 0) {
                    dp[i][a] = dp[i - 1][a];
                } else {
                    dp[i][a] = dp[i - 1][a] + dp[i][remain];
                }
            }
        }
        
        return (int)dp[n][amount];
    }
};

int main() {
    Solution sol;
    vector<int> coins = {1, 2, 5};
    int amount = 5;
    cout << sol.change(amount, coins) << endl; // Expected output: 4
    return 0;
}


// g++ -std=c++17 Leetcode_0518_Coin_Change_II.cpp -o test