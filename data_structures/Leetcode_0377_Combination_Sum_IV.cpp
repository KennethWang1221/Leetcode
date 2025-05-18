#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        vector<unsigned long long> dp(target + 1, 0);
        dp[0] = 1; // One way to make sum 0: choose nothing

        for (int j = 1; j <= target; ++j) {
            for (int num : nums) {
                if (j >= num) {
                    dp[j] += dp[j - num];
                }
            }
        }

        return dp[target];
    }
};

// Test Case
int main() {
    Solution sol;
    vector<int> nums = {1, 2, 3};
    int target = 4;
    cout << sol.combinationSum4(nums, target) << endl; // Output: 7
    return 0;
}


// g++ -std=c++17 Leetcode_0377_Combination_Sum_IV.cpp -o test