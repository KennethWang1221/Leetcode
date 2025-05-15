#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> comb;
        vector<bool> used(nums.size(), false);  // Track used elements
        sort(nums.begin(), nums.end());  // Sort to handle duplicates
        backtracking(nums, comb, used, res);
        return res;
    }

private:
    void backtracking(vector<int>& nums, vector<int>& comb, vector<bool>& used, vector<vector<int>>& res) {
        if (comb.size() == nums.size()) {
            res.push_back(comb);  // If the combination size matches, add to result
            return;
        }

        for (int i = 0; i < nums.size(); ++i) {
            // Skip if the number is already used or if it's a duplicate in the current recursion
            if (used[i] || (i > 0 && nums[i] == nums[i - 1] && !used[i - 1])) {
                continue;
            }

            used[i] = true;  // Mark the current number as used
            comb.push_back(nums[i]);  // Add the number to the current combination
            backtracking(nums, comb, used, res);  // Recurse to build the next combination
            comb.pop_back();  // Backtrack by removing the last added number
            used[i] = false;  // Unmark the number as used
        }
    }
};

int main() {
    Solution solution;
    vector<int> nums = {1, 1, 2};

    vector<vector<int>> result = solution.permuteUnique(nums);

    // Print the result
    for (const auto& perm : result) {
        cout << "[ ";
        for (int num : perm) {
            cout << num << " ";
        }
        cout << "]" << endl;
    }

    return 0;
}


// g++ -std=c++17 Leetcode_0047_Permutations_II.cpp -o test