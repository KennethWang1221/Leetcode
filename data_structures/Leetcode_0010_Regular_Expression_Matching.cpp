#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    bool isMatch(string s, string p) {
        int m = s.size(), n = p.size();
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
        dp[0][0] = true; // Empty string matches empty pattern

        // Fill in first row (s = "")
        for (int j = 2; j <= n; ++j)
            if (p[j - 1] == '*') dp[0][j] = dp[0][j - 2];

        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p[j - 1] == '*') {
                    // Option 1: skip the previous char
                    dp[i][j] = dp[i][j - 2];
                    // Option 2: if current char matches, reuse previous result
                    if (!dp[i][j] && (p[j - 2] == '.' || p[j - 2] == s[i - 1]))
                        dp[i][j] = dp[i - 1][j];
                } else {
                    // Normal char or '.'
                    if (p[j - 1] == '.' || p[j - 1] == s[i - 1])
                        dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }

        return dp[m][n];
    }
};

// Test Cases
int main() {
    Solution sol;
    cout << boolalpha;
    cout << "Test 1: " << sol.isMatch("aa", "a") << endl;       // false
    cout << "Test 2: " << sol.isMatch("aa", "a*") << endl;      // true
    cout << "Test 3: " << sol.isMatch("ab", ".*") << endl;      // true
    cout << "Test 4: " << sol.isMatch("aab", "c*a*b") << endl;  // true
    cout << "Test 5: " << sol.isMatch("mississippi", "mis*is*p*.") << endl; // false
    return 0;
}

// g++ -std=c++17 Leetcode_0010_Regular_Expression_Matching.cpp -o test  