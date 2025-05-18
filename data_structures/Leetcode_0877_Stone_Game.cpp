#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

using namespace std;

class Solution {
public:
    unordered_map<string, int> dp;

    // Generate unique key for memoization
    string getKey(int l, int r) {
        return to_string(l) + "," + to_string(r);
    }

    int dfs(int l, int r, vector<int>& piles) {
        if (l > r) return 0;
        string key = getKey(l, r);
        if (dp.count(key)) return dp[key];

        bool evenTurn = ((r - l) % 2 == 0); // Alex's turn if even length of remaining section
        int left = evenTurn ? piles[l] : 0;
        int right = evenTurn ? piles[r] : 0;

        int pickLeft = dfs(l + 1, r, piles) + left;
        int pickRight = dfs(l, r - 1, piles) + right;

        dp[key] = max(pickLeft, pickRight);
        return dp[key];
    }

    bool stoneGame(vector<int>& piles) {
        int n = piles.size();
        int total = 0;
        for (int x : piles) total += x;

        int alexScore = dfs(0, n - 1, piles);
        return alexScore > total / 2;
    }
};

// Test Case
int main() {
    Solution sol;
    vector<int> piles = {5, 3, 4, 5};
    cout << boolalpha << sol.stoneGame(piles) << endl; // Output: true
    return 0;
}

// g++ -std=c++17 Leetcode_0877_Stone_Game.cpp -o test 