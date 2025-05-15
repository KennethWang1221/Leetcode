#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    bool makesquare(vector<int>& matchsticks) {
        int total = 0;
        for (int m : matchsticks) {
            total += m;
        }

        // If the total length is not divisible by 4, return false
        if (total % 4 != 0) {
            return false;
        }

        int length = total / 4;
        vector<int> sides(4, 0);

        // Sort matchsticks in reverse order to optimize backtracking
        sort(matchsticks.rbegin(), matchsticks.rend());

        return backtrack(matchsticks, sides, 0, length);
    }

private:
    bool backtrack(vector<int>& matchsticks, vector<int>& sides, int i, int length) {
        if (i == matchsticks.size()) {
            // If all matchsticks have been placed, check if all sides are equal to the target length
            return sides[0] == length && sides[1] == length && sides[2] == length && sides[3] == length;
        }

        for (int j = 0; j < 4; ++j) {
            // Try placing the matchstick on each side if it fits
            if (sides[j] + matchsticks[i] <= length) {
                sides[j] += matchsticks[i];
                if (backtrack(matchsticks, sides, i + 1, length)) {
                    return true;
                }
                sides[j] -= matchsticks[i];  // Backtrack
            }
        }
        return false;
    }
};

int main() {
    Solution solution;

    vector<int> matchsticks1 = {1, 1, 2, 2, 2};
    vector<int> matchsticks2 = {3, 3, 3, 3, 4};

    bool result1 = solution.makesquare(matchsticks1);
    bool result2 = solution.makesquare(matchsticks2);

    cout << (result1 ? "True" : "False") << endl;  // Output: True
    cout << (result2 ? "True" : "False") << endl;  // Output: False

    return 0;
}

// g++ -std=c++17 Leetcode_0473_Matchsticks_to_Square.cpp -o test