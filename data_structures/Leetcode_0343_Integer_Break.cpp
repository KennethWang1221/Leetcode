#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int integerBreak(int n) {
        vector<int> dp(n + 1, 0);
        dp[1] = 1;

        for (int num = 2; num <= n; ++num) {
            // For n itself, we must split into at least two parts, so don't allow num itself
            dp[num] = (num == n) ? 0 : num;

            for (int i = 1; i < num; ++i) {
                int j = num - i;
                dp[num] = max(dp[num], dp[i] * dp[j]);
            }
        }

        return dp[n];
    }
};

// Test case
int main() {
    Solution sol;
    cout << sol.integerBreak(4) << endl;  // Output: 4
    return 0;
}
// g++ -std=c++17 Leetcode_0343_Integer_Break.cpp -o test  