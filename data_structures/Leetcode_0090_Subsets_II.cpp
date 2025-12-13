#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> comb;
        sort(nums.begin(), nums.end());  // Sort to handle duplicates
        backtracking2(nums, 0, comb, res);
        return res;
    }

private:
    void backtracking(vector<int>& nums, int start, vector<int>& comb, vector<vector<int>>& res) {
        res.push_back(comb);  // Always add the current combination to the result
        
        for (int i = start; i < nums.size(); ++i) {
            // Skip duplicates by ensuring that the current element is not the same as the previous one
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }

            comb.push_back(nums[i]);  // Include the current number in the combination
            backtracking(nums, i + 1, comb, res);  // Recurse with the next index
            comb.pop_back();  // Backtrack to explore other combinations
        }
    }

    void backtracking2(vector<int>& nums, int start, vector<int>& comb, vector<vector<int>>& res) {
        int n = nums.size();
        if (comb.size() <= n && find(res.begin(), res.end(), comb) == res.end()){
            res.push_back(comb);
        }
        for (int i = start; i < nums.size(); ++i) {
            // Skip duplicates by ensuring that the current element is not the same as the previous one
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }

            comb.push_back(nums[i]);  // Include the current number in the combination
            backtracking(nums, i + 1, comb, res);  // Recurse with the next index
            comb.pop_back();  // Backtrack to explore other combinations
        }
    }
};

int main() {
    Solution solution;
    vector<int> nums = {1, 2, 2};
    vector<int> nums2 = {4, 4, 4, 1, 4};

    vector<vector<int>> result = solution.subsetsWithDup(nums);
    cout << "Subsets for {1, 2, 2}:" << endl;
    for (const auto& comb : result) {
        cout << "[ ";
        for (int num : comb) {
            cout << num << " ";
        }
        cout << "]" << endl;
    }

    result = solution.subsetsWithDup(nums2);
    cout << "Subsets for {4, 4, 4, 1, 4}:" << endl;
    for (const auto& comb : result) {
        cout << "[ ";
        for (int num : comb) {
            cout << num << " ";
        }
        cout << "]" << endl;
    }

    return 0;
}


// g++ -std=c++17 Leetcode_0090_Subsets_II.cpp -o test