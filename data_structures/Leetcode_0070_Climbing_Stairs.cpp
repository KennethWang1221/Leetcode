#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    int climbStairs(int n) {
        /**
         * static:  Used in classes/structs
         * Means: "This variable belongs to the class itself, not to any specific object "
         * Shared among all instances of the class
         * static const vector<int> steps = {1,2};
         * It means:
         * There is one shared steps vector for all instances of Solution
         * It cannot be modified (because it's const)
         * This is fine — but only necessary if:
         * You have multiple calls to climbStairs
         * And want to avoid reinitializing steps every time
         */
        static const vector<int> steps = {1,2};
        vector<int> dp(n + 1, 0);
        dp[0] = 1;

        // Inefficient Initialization of dp: DO not use:
        // vector<int> dp = {};
        // for (int i = 0; i < n+1; i++){
        //     dp.push_back(0);
        // } 
        

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


/*
step = [1,2]
    0 1 2 3
1   1 1 1 2
2   1 1 2 3
*/

// g++ -std=c++17 Leetcode_0070_Climbing_Stairs.cpp -o test