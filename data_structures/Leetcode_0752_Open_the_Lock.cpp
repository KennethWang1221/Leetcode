#include <iostream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <string>

using namespace std;

class Solution {
public:
    // Function to get all possible next states for a given wheel
    vector<string> children(const string& wheel) {
        vector<string> res;
        for (int i = 0; i < 4; i++) {
            // Increment the current digit
            string next_wheel = wheel;
            next_wheel[i] = (wheel[i] - '0' + 1) % 10 + '0';
            res.push_back(next_wheel);

            // Decrement the current digit
            next_wheel = wheel;
            next_wheel[i] = (wheel[i] - '0' + 9) % 10 + '0';  // Same as wheel[i] - 1, mod 10
            res.push_back(next_wheel);
        }
        return res;
    }

    int openLock(vector<string>& deadends, string target) {
        if (find(deadends.begin(), deadends.end(), "0000") != deadends.end()) {
            return -1;  // If "0000" is a deadend, return -1
        }

        unordered_set<string> visit(deadends.begin(), deadends.end());  // Mark all deadends as visited
        queue<pair<string, int>> q;  // Queue to store the state of the wheel and the number of turns
        q.push({"0000", 0});

        while (!q.empty()) {
            auto [wheel, turns] = q.front();
            q.pop();

            if (wheel == target) {
                return turns;  // If the target is found, return the number of turns
            }

            for (const string& child : children(wheel)) {
                if (visit.find(child) == visit.end()) {
                    visit.insert(child);
                    q.push({child, turns + 1});
                }
            }
        }

        return -1;  // If we cannot reach the target, return -1
    }
};

int main() {
    Solution sol;
    vector<string> deadends = {"0201", "0101", "0102", "1212", "2002"};
    string target = "0202";
    int res = sol.openLock(deadends, target);
    cout << "Result: " << res << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0752_Open_the_Lock.cpp -o test