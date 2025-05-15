#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    int findJudge(int n, vector<vector<int>>& trust) {
        if (n == 1 && trust.empty()) return 1;

        unordered_map<int, int> premap;

        // Initialize the premap for each person
        for (int i = 1; i <= n; ++i) {
            premap[i] = 0;
        }

        // Process trust pairs
        for (const auto& t : trust) {
            int src = t[0], dst = t[1];
            premap[src] -= 1;  // src trusts someone, so they can't be the judge
            premap[dst] += 1;  // dst is trusted, could be the judge
        }

        // Find the person who is trusted by everyone else (value == n-1)
        for (const auto& entry : premap) {
            if (entry.second == n - 1) {
                return entry.first;
            }
        }

        return -1;  // No judge found
    }
};

int main() {
    Solution solution;

    // Test case 1
    vector<vector<int>> trust1 = {{1, 3}, {2, 3}};
    cout << solution.findJudge(3, trust1) << endl;  // Expected output: 3

    // Test case 2
    vector<vector<int>> trust2 = {{1, 3}, {2, 3}, {3, 1}};
    cout << solution.findJudge(3, trust2) << endl;  // Expected output: -1

    return 0;
}


// g++ -std=c++17 Leetcode_0997_Find_the_Town_Judge.cpp -o test