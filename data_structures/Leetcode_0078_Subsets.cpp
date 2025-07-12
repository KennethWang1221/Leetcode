#include <iostream>
#include <unordered_set>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res = {};
        vector<int> comb = {};
        int start = 0;
        backtrack(nums, start, comb, res);
        return res;
    }

private:
    void backtrack(vector<int>& nums, int start, vector<int>& comb, vector<vector<int>>& res) {
        // Check if the current combination is not already in the result
        if (find(res.begin(), res.end(), comb) == res.end()){
            res.push_back(comb);
        }
        
        if (comb.size() > nums.size()){
            return;
        }
        // Explore further elements
        for (int i = start; i < nums.size(); ++i) {
            comb.push_back(nums[i]);
            backtrack(nums, i + 1, comb, res);
            comb.pop_back();
        }
    }
};

int main() {
    Solution sol;
    vector<int> nums = {1, 2, 3};
    vector<vector<int>> result = sol.subsets(nums);
    // Print the result
    cout << "[";
    for (size_t i = 0; i < result.size(); ++i) {
        cout << "[";
        for (size_t j = 0; j < result[i].size(); ++j) {
            cout << result[i][j];
            if (j < result[i].size() - 1) cout << ",";
        }
        cout << "]";
        if (i < result.size() - 1) cout << ", ";
    }
    cout << "]" << endl;


    // another way
    for (const vector<int>& subset : result) {
        cout << "{ ";
        for (int num : subset) {
            cout << num << " ";
        }
        cout << "}\n";
    }

    // third way
    for (const auto& subset : result) {
        cout << "{ ";
        for (int num : subset) {
            cout << num << " ";
        }
        cout << "}\n";
    }

    return 0;
}

// g++ -std=c++17 Leetcode_0078_Subsets.cpp -o test