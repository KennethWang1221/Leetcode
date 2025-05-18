#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <climits>

using namespace std;

class Solution {
public:
    unordered_map<string, int> dp;

    // Helper function to generate unique key for memoization
    string getKey(bool alice, int i, int M) {
        return to_string(alice) + "," + to_string(i) + "," + to_string(M);
    }

    int dfs(bool alice, vector<int>& piles, int i, int M) {
        int n = piles.size();
        if (i == n) return 0;

        string key = getKey(alice, i, M);
        if (dp.count(key)) return dp[key];

        int res = alice ? 0 : INT_MAX;
        int total = 0;

        for (int X = 1; X <= 2 * M; ++X) {
            if (i + X > n) break;
            total += piles[i + X - 1];
            if (alice) {
                res = max(res, total + dfs(false, piles, i + X, max(M, X)));
            } else {
                res = min(res, dfs(true, piles, i + X, max(M, X)));
            }
        }

        dp[key] = res;
        return res;
    }

    int stoneGameII(vector<int>& piles) {
        return dfs(true, piles, 0, 1);
    }
};

// Test Case
int main() {
    Solution sol;
    vector<int> piles = {2, 7, 9, 4, 4};
    cout << sol.stoneGameII(piles) << endl; // Output: 10
    return 0;
}

// g++ -std=c++17 Leetcode_1140_Stone_Game_II.cpp -o test 