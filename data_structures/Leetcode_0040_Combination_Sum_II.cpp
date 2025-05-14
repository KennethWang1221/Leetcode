#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> comb;
        sort(candidates.begin(), candidates.end());  // Sort to handle duplicates efficiently
        backtracking(candidates, target, 0, comb, 0, res);
        return res;
    }

private:
    void backtracking(vector<int>& candidates, int target, int start, vector<int>& comb, int total, vector<vector<int>>& res) {
        // If the current total equals the target, add the combination to the result
        if (total == target) {
            res.push_back(comb);
            return;
        }
        
        // If the total exceeds the target or we've considered all candidates, stop
        if (total > target || start >= candidates.size()) {
            return;
        }

        for (int i = start; i < candidates.size(); ++i) {
            // Skip duplicates
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }

            // Add the current candidate to the combination and backtrack
            comb.push_back(candidates[i]);
            backtracking(candidates, target, i + 1, comb, total + candidates[i], res);
            comb.pop_back();  // Remove the last element to try the next candidate
        }
    }
};

int main() {
    Solution solution;
    vector<int> candidates = {10, 1, 2, 7, 6, 1, 5};
    int target = 8;

    vector<vector<int>> result = solution.combinationSum2(candidates, target);

    // Print the result
    for (const auto& comb : result) {
        cout << "[ ";
        for (int num : comb) {
            cout << num << " ";
        }
        cout << "]" << endl;
    }

    return 0;
}
// g++ -std=c++17 Leetcode_0040_Combination_Sum_II.cpp -o test