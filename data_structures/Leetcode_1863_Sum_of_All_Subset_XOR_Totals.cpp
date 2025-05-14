#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    int subsetXORSum(vector<int>& nums) {
        int n = nums.size();
        
        // Recursive DFS function to compute XOR sum of subsets
        return dfs(nums, 0, 0);
    }

private:
    int dfs(const vector<int>& nums, int i, int total) {
        // If we have considered all elements
        if (i == nums.size()) {
            return total;
        }

        // Consider both possibilities: include or exclude the current number
        return dfs(nums, i + 1, total ^ nums[i]) + dfs(nums, i + 1, total);
    }
};

int main() {
    Solution solution;
    vector<int> nums = {5, 1, 6};
    cout << solution.subsetXORSum(nums) << endl;  // Expected output: 28

    return 0;
}


// g++ -std=c++17 Leetcode_1863_Sum_of_All_Subset_XOR_Totals.cpp -o test