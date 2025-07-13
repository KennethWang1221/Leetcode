#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    int climbStairs(int n) {
        static const vector<int> steps = {1,2};
        // vector<int> dp(n + 1, 0);

        vector<int> dp = {};
        for (int i = 0; i < n+1; i++){
            dp.push_back(0);
        }
        dp[0] = 1;

        for (int i = 1; i < n+1; i++){
            for (const int& step : steps){
                int remain = i - step;
                if (remain < 0){
                    dp[i] = dp[i];
                } else{
                    dp[i] = dp[remain] + dp[i];
                }
            }
        }
        
        return dp[n];
    }
};

int main() {
    Solution sol;
    int n = 3;
    int result = sol.climbStairs(n);
    cout << "Test case n=3: " << result << endl; // Output should be 3
    return 0;
}

// g++ -std=c++17 Leetcode_0070_Climbing_Stairs.cpp -o test