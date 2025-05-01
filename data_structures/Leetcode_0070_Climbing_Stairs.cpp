#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    int climbStairs(int n) {
        vector<int> nums = {1, 2};
        vector<int> dp(n + 1, 0);
        dp[0] = 1; // Base case: 1 way to stay at ground (do nothing)
        
        for (int a = 1; a <= n; ++a) {
            for (int step : nums) {
                int remain = a - step;
                if (remain >= 0) {
                    dp[a] += dp[remain];
                }
            }
        }
        
        return dp[n];
    }
};

int main() {
    Solution sol;
    int result = sol.climbStairs(3);
    cout << "Test case n=3: " << result << endl; // Output should be 3
    return 0;
}

// g++ -std=c++17 Leetcode_0070_Climbing_Stairs.cpp -o test