#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    int uniquePaths(int m, int n) {
        // Initialize dp with (m x n), all values initially 0
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        dp[m-1][n] = 1;

        for (int i = m-1; i > -1; i--){
            for (int j = n-1; j > -1; j--){
                dp[i][j] = dp[i+1][j] + dp[i][j+1];
            }
        }

        return dp[0][0];
    }

};

// Test Case
int main() {
    int m = 3;
    int n = 7;
    Solution s;
    int res = 0;
    res = s.uniquePaths(m,n);
    cout << res << endl;
    return 0;
}

// g++ -std=c++17 Leetcode_0062_Unique_Paths.cpp -o test     