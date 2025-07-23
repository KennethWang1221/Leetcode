#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> res = {};
        int total = 0;
        vector<int> comb = {};
        int start = 0;
        backtracking(candidates, target, total, comb, res, start);
        return res;
    }

private:
    void backtracking(vector<int>& candidates, int target, int total, vector<int>& comb, vector<vector<int>>& res, int start) {
        // If the current total equals the target, add the combination to the result
        if (total == target) {
            res.push_back(comb);
            return;
        }
        
        // If the total exceeds the target or we've considered all candidates, stop
        if (total > target || start >= candidates.size()) {
            return;
        }

        for (int i = start; i < candidates.size(); i++){
            comb.push_back(candidates[i]);
            total += candidates[i];
            backtracking(candidates, target, total, comb, res, i);
            comb.pop_back();
            total -= candidates[i];
        }
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


    for (const vector<int>& subset: result){
        for (int num: subset){
            cout << num << endl;
        }

    }

    return 0;
}


// g++ -std=c++17 Leetcode_0039_Combination_Sum.cpp -o test && ./test