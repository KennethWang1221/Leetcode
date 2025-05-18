#include <iostream>
#include <vector>
#include <string>

using namespace std;

#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    int numDistinct(string s, string t) {
        int m = s.length();
        int n = t.length();

        // Use unsigned long long to reduce risk of overflow
        vector<vector<unsigned long long>> dp(n + 1, vector<unsigned long long>(m + 1, 0));

        // Base case: empty t can be formed once from any s
        for (int j = 0; j <= m; ++j) {
            dp[n][j] = 1;
        }

        // Fill DP table bottom-up
        for (int i = n - 1; i >= 0; --i) {
            for (int j = m - 1; j >= 0; --j) {
                if (t[i] == s[j]) {
                    dp[i][j] = dp[i + 1][j + 1] + dp[i][j + 1];
                } else {
                    dp[i][j] = dp[i][j + 1];
                }
            }
        }

        return dp[0][0]; // Safe cast to int since LeetCode expects small final result
    }
};

// Test Case
int main() {
    Solution sol;
    string s = "rabbbit";
    string t = "rabbit";
    cout << sol.numDistinct(s, t) << endl; // Output: 3
    return 0;
}

// g++ -std=c++17 Leetcode_0115_Distinct_Subsequences.cpp -o test      