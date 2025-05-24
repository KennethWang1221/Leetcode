#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    bool isSubsequence(string s, string t) {
        int m = s.size();
        int n = t.size();

        // dp[i][j] = true if s[0..i-1] is subsequence of t[0..j-1]
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));

        // Base case: empty s is always a subsequence of any t
        for (int j = 0; j <= n; ++j)
            dp[0][j] = true;

        // Fill the DP table
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (s[i - 1] == t[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = dp[i][j - 1]; // skip current char in t
                }
            }
        }

        return dp[m][n];
    }
};

// Test Case
int main() {
    Solution sol;
    cout << boolalpha << sol.isSubsequence("axc", "ahbgdc") << endl; // Output: false
    cout << boolalpha << sol.isSubsequence("abc", "ahbgdc") << endl; // Output: true
    return 0;
}

// g++ -std=c++17 Leetcode_0392_Is_Subsequence.cpp -o test