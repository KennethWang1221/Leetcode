#include <iostream>   // for cout, endl
#include <string>     // for string
#include <vector>     // for vector
#include <algorithm>  // for min

using namespace std;  // This allows us to write `cout` instead of `std::cout`, etc.

class Solution {
public:
    int minDistance(string word1, string word2) {
        int m = word1.size();
        int n = word2.size();

        vector<vector<int>> dp(m + 1, vector<int>(n + 1));

        // Initialize base cases
        for (int i = 0; i <= m; ++i)
            dp[i][0] = i;
        for (int j = 0; j <= n; ++j)
            dp[0][j] = j;

        // Fill DP table
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (word1[i - 1] == word2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + min({
                        dp[i - 1][j],       // Delete
                        dp[i][j - 1],       // Insert
                        dp[i - 1][j - 1]    // Replace
                    });
                }
            }
        }

        return dp[m][n];
    }
};

// Test case
int main() {
    Solution sol;
    string word1 = "horse";
    string word2 = "ros";
    cout << sol.minDistance(word1, word2) << endl; // Output: 3 ✅
    return 0;
}