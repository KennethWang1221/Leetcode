#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> comb;
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

        // Include the current number and backtrack
        comb.push_back(candidates[start]);
        backtracking(candidates, target, start, comb, total + candidates[start], res); // not moving to the next element
        comb.pop_back(); // Backtrack to try other possibilities

        // Exclude the current number and move to the next element
        backtracking(candidates, target, start + 1, comb, total, res);
    }
};

int main() {
    Solution solution;
    vector<int> candidates = {2, 3, 6, 7};
    int target = 7;
    
    vector<vector<int>> result = solution.combinationSum(candidates, target);
    
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


// g++ -std=c++17 Leetcode_0039_Combination_Sum.cpp -o test