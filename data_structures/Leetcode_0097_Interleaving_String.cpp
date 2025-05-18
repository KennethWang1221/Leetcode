#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    bool isInterleave(string s1, string s2, string s3) {
        int s1_n = s1.size();
        int s2_n = s2.size();
        int s3_n = s3.size();

        // If the lengths of s1 and s2 do not add up to the length of s3, return false
        if (s1_n + s2_n != s3_n) {
            return false;
        }

        // Initialize DP table with false values
        vector<vector<bool>> dp(s1_n + 1, vector<bool>(s2_n + 1, false));
        dp[s1_n][s2_n] = true;  // Base case: empty s1 and s2 form an empty s3

        // Fill the DP table from bottom right to top left
        for (int i = s1_n; i >= 0; --i) {
            for (int j = s2_n; j >= 0; --j) {
                // Check if we can match the current character of s1 with s3
                if (i < s1_n && s1[i] == s3[i + j] && dp[i + 1][j]) {
                    dp[i][j] = true;
                }
                // Check if we can match the current character of s2 with s3
                if (j < s2_n && s2[j] == s3[i + j] && dp[i][j + 1]) {
                    dp[i][j] = true;
                }
            }
        }

        return dp[0][0];  // The result is at dp[0][0], meaning can we interleave from the start of s1 and s2 to form s3
    }
};

int main() {
    Solution sol;
    string s1 = "aabcc";
    string s2 = "dbbca";
    string s3 = "aadbbcbcac";
    
    bool result = sol.isInterleave(s1, s2, s3);
    cout << (result ? "True" : "False") << endl;  // Expected output: True

    return 0;
}


// g++ -std=c++17 Leetcode_0097_Interleaving_String.cpp -o test 