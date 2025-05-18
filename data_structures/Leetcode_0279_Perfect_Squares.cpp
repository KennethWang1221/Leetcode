#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

class Solution {
public:
    int numSquares(int n) {
        // Step 1: Precompute the list of perfect squares less than or equal to n
        vector<int> nums;
        int num = 1;
        while (num * num <= n) {
            nums.push_back(num * num);
            num++;
        }

        // Step 2: Initialize dp array where dp[i] represents the minimum number of perfect squares that sum up to i
        vector<int> dp(n + 1, INT_MAX);  // Fill dp with a large number initially
        dp[0] = 0;  // Base case: 0 can be formed with 0 squares

        // Step 3: Dynamic programming to fill the dp array
        for (int i = 0; i < nums.size(); ++i) {
            for (int j = nums[i]; j <= n; ++j) {
                dp[j] = min(dp[j], dp[j - nums[i]] + 1);  // Take the minimum of the current value and the value using this square
            }
        }

        // Step 4: Return the result, which is the minimum number of squares to sum up to n
        return dp[n];
    }
};

int main() {
    Solution sol;
    int n = 12;
    cout << sol.numSquares(n) << endl;  // Expected output: 3 (since 12 = 4 + 4 + 4)
    
    return 0;
}


// g++ -std=c++17 Leetcode_0279_Perfect_Squares.cpp -o test   