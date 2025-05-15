#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> res;
        vector<int> comb;
        backtracking(n, k, 1, comb, res);
        return res;
    }

private:
    void backtracking(int n, int k, int start, vector<int>& comb, vector<vector<int>>& res) {
        // If the combination size equals k, add it to the result
        if (comb.size() == k) {
            res.push_back(comb);
            return;
        }

        // Explore further combinations by choosing each number from start to n
        for (int i = start; i <= n; ++i) {
            comb.push_back(i);  // Choose the current number
            backtracking(n, k, i + 1, comb, res);  // Recurse with the next start value
            comb.pop_back();  // Backtrack to explore other combinations
        }
    }
};

int main() {
    Solution solution;
    int n = 4, k = 2;

    vector<vector<int>> result = solution.combine(n, k);

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


// g++ -std=c++17 Leetcode_0077_Combinations.cpp -o test