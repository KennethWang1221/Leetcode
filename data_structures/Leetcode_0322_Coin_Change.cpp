#include <vector>
#include <climits>
#include <algorithm>
#include <iostream>

using namespace std;

class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        int n = coins.size();
        vector<vector<int>> dp(n+1, vector<int>(amount+1, INT_MAX));

        for (int i=0;i<n+1;i++){
            dp[i][0] = 0;
        }

        for (int i=1;i<n+1;i++){
            for (int j=1;j<amount+1;j++){
                int remain = j - coins[i-1];
                if(remain < 0){
                    dp[i][j] = dp[i-1][j];
                }else{
                    if (dp[i][remain] == INT_MAX){
                        dp[i][j] = dp[i-1][j];
                    }else{
                        dp[i][j] = min(1 + dp[i][remain], dp[i-1][j]);
                    }
                    
                }
            }
        }
        
        if (dp[n][amount] != INT_MAX){
            return dp[n][amount];
        }else{
            return -1;
        }
        
    }
};

int main() {
    Solution sol;
    vector<int> coins = {1, 2, 5};
    int amount = 11;
    cout << sol.coinChange(coins, amount) << endl; // Expected output: 3
    return 0;
}


// g++ -std=c++17 Leetcode_0322_Coin_Change.cpp -o test