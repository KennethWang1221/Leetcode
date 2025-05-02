#include <iostream>
#include <unordered_set>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> comb;
        backtrack(nums, 0, comb, res);
        return res;
    }

private:
    void backtrack(vector<int>& nums, int start, vector<int>& comb, vector<vector<int>>& res) {
        // Check if the current combination is not already in the result
        if (!contains(res, comb)) {
            res.push_back(comb);
        }
        // Explore further elements
        for (int i = start; i < nums.size(); ++i) {
            comb.push_back(nums[i]);
            backtrack(nums, i + 1, comb, res);
            comb.pop_back();
        }
    }

    bool contains(const vector<vector<int>>& res, const vector<int>& comb) {
        return find(res.begin(), res.end(), comb) != res.end();
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
    return 0;
}

// g++ -std=c++17 Leetcode_0078_Subsets.cpp -o test