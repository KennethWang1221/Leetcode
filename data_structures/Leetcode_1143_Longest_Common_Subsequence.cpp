#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int n1 = text1.size();
        int n2 = text2.size();

        // Create a 2D dp array with dimensions (n1+1) x (n2+1), initialized to 0
        vector<vector<int>> dp(n1 + 1, vector<int>(n2 + 1, 0));

        // Fill the dp array using dynamic programming
        for (int i = n1 - 1; i >= 0; --i) {
            for (int j = n2 - 1; j >= 0; --j) {
                if (text1[i] == text2[j]) {
                    dp[i][j] = 1 + dp[i + 1][j + 1];  // Characters match, add 1 to the result
                } else {
                    dp[i][j] = max(dp[i][j + 1], dp[i + 1][j]);  // Take the maximum of the two possibilities
                }
            }
        }

        return dp[0][0];  // Return the result stored in dp[0][0]
    }
};

int main() {
    Solution sol;

    string text1 = "bsbininm";
    string text2 = "jmjkbkjkv";
    int result = sol.longestCommonSubsequence(text1, text2);
    cout << result << endl;  // Expected output: 1 (Longest common subsequence: "b")

    return 0;
}


// g++ -std=c++17 Leetcode_1143_Longest_Common_Subsequence.cpp -o test  