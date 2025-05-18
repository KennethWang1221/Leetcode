#include <iostream>
#include <vector>
#include <string>
#include <climits>

using namespace std;

class Solution {
public:
    string stoneGameIII(vector<int>& stoneValue) {
        int n = stoneValue.size();
        vector<long long> dp(n + 4, INT_MIN); // Extra space to avoid index issues
        dp[n] = 0; // Base case: no stones left means no score difference

        for (int i = n - 1; i >= 0; --i) {
            long long sum = 0;
            for (int k = 0; k < 3 && i + k < n; ++k) {
                sum += stoneValue[i + k];
                dp[i] = max(dp[i], sum - dp[i + k + 1]);
            }
        }

        if (dp[0] > 0) return "Alice";
        else if (dp[0] < 0) return "Bob";
        else return "Tie";
    }
};

// Test Cases
int main() {
    Solution sol;

    vector<int> stones1 = {1, 2, 3, 7};
    cout << "Test Case 1: " << sol.stoneGameIII(stones1) << endl; // Output: Bob

    vector<int> stones2 = {1, 2, 3, -9};
    cout << "Test Case 2: " << sol.stoneGameIII(stones2) << endl; // Output: Alice

    vector<int> stones3 = {1, 2, 3, 4, 5};
    cout << "Test Case 3: " << sol.stoneGameIII(stones3) << endl; // Output: Alice

    vector<int> stones4 = {0};
    cout << "Test Case 4: " << sol.stoneGameIII(stones4) << endl; // Output: Tie

    vector<int> stones5 = {-1, -2, -3};
    cout << "Test Case 5: " << sol.stoneGameIII(stones5) << endl; // Output: Bob

    return 0;
}

// g++ -std=c++17 Leetcode_1406_Stone_Game_III.cpp -o test 