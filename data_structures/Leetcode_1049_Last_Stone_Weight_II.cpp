#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int lastStoneWeightII(vector<int>& stones) {
        int total_sum = 0;
        for (int stone : stones) {
            total_sum += stone;
        }

        int target = total_sum / 2;
        int n = stones.size();

        // dp[i][j] = whether we can form a sum j using first i stones
        vector<vector<bool>> dp(n + 1, vector<bool>(target + 1, false));

        // Base case: sum 0 can always be formed
        for (int i = 0; i <= n; ++i) {
            dp[i][0] = true;
        }

        for (int i = 1; i <= n; ++i) {
            int stone = stones[i - 1];
            for (int j = 1; j <= target; ++j) {
                if (stone > j) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j] || dp[i - 1][j - stone];
                }
            }
        }

        // Find the largest j where dp[n][j] == true
        int max_weight = 0;
        for (int j = target; j >= 0; --j) {
            if (dp[n][j]) {
                max_weight = j;
                break;
            }
        }

        return total_sum - 2 * max_weight;
    }
};

// Test Case
int main() {
    Solution sol;
    vector<int> stones = {2,7,4,1,8,1};
    cout << sol.lastStoneWeightII(stones) << endl; // Output: 1
    return 0;
}

// g++ -std=c++17 Leetcode_1049_Last_Stone_Weight_II.cpp -o test  