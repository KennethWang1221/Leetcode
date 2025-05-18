#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    int uniquePaths(int m, int n) {
        // Initialize DP table with zeros
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

        // Base case: bottom-right cell
        dp[m - 1][n - 1] = 1;

        // Fill DP table from bottom to top, right to left
        for (int i = m - 1; i >= 0; --i) {
            for (int j = n - 1; j >= 0; --j) {
                if (i == m - 1 && j == n - 1) continue; // Skip base case
                dp[i][j] = dp[i + 1][j] + dp[i][j + 1];
            }
        }

        return dp[0][0];
    }
};

// Test Case
int main() {
    Solution sol;
    int m = 3;
    int n = 7;
    cout << sol.uniquePaths(m, n) << endl; // Output: 28
    return 0;
}

// g++ -std=c++17 Leetcode_0062_Unique_Paths.cpp -o test     