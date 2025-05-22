#include <iostream>
#include <vector>
#include <queue>
#include <string>

using namespace std;

class Solution {
public:
    string predictPartyVictory(string senate) {
        int n = senate.size();
        queue<int> radiant, dire;

        // Store indices of each senator by party
        for (int i = 0; i < n; ++i) {
            if (senate[i] == 'R')
                radiant.push(i);
            else
                dire.push(i);
        }

        // Simulate the rounds
        while (!radiant.empty() && !dire.empty()) {
            int r = radiant.front(); radiant.pop();
            int d = dire.front(); dire.pop();

            // The senator whose turn comes earlier bans the other
            if (r < d) {
                radiant.push(r + n); // Re-add with new "turn"
            } else {
                dire.push(d + n);
            }
        }

        return radiant.empty() ? "Dire" : "Radiant";
    }
};

// Test Case
int main() {
    Solution sol;
    string senate = "RDPRDRDDRDRDRDDRRRDRDRDRDDRDRDRDDRDRDRDDRDRDRDRRDRDDRRDRDRDRDRDRDRDRRDRDRDRDRDRDRDRDDRRDRDRDRDDRDRDRDR";
    cout << sol.predictPartyVictory(senate) << endl; // Output: "Radiant"

    string senate2 = "RDD";
    cout << sol.predictPartyVictory(senate2) << endl; // Output: "Radiant"

    string senate3 = "DDRRR";
    cout << sol.predictPartyVictory(senate3) << endl; // Output: "Dire"

    return 0;
}
// g++ -std=c++11 Leetcode_0649_Dota2Senate.cpp -o test && ./test