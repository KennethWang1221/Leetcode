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



    int climbStairs_fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
    
        vector<int> dp(n + 1, 0);
        dp[1] = 1;
        dp[2] = 2;
    
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
    
        return dp[n];
    }
};

int main() {
    Solution sol;
    int n = 3;
    int result = sol.climbStairs(n);
    int result2 = sol.climbStairs_fibonacci(n);
    cout << "Test case n=3: " << result << endl; // Output should be 3
    cout << "Test case n=3: " << result2 << endl; // Output should be 3
    return 0;
}


/*
Below matrix is totally wrong!
step = [1,2]
    0 1 2 3
1   1 1 1 2
2   1 1 2 3

correct version: 

    1 2 
0   1  
1   1 1
2   1 2
3   2 3 

start          [1, 0, 0, 0]

i=1, step=1    [1, 1, 0, 0]
i=1, step=2    [1, 1, 0, 0]

i=2, step=1    [1, 1, 1, 0]
i=2, step=2    [1, 1, 2, 0]

i=3, step=1    [1, 1, 2, 2]
i=3, step=2    [1, 1, 2, 3]

*/

// g++ -std=c++17 Leetcode_0070_Climbing_Stairs.cpp -o test