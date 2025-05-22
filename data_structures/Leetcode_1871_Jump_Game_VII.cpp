#include <iostream>
#include <string>
#include <queue>

using namespace std;

class Solution {
public:
    bool canReach(string s, int minJump, int maxJump) {
        int n = s.size();
        queue<int> q;
        q.push(0);
        int farthest = 0;

        while (!q.empty()) {
            int i = q.front(); q.pop();

            // Start from the next index after previous farthest
            int start = max(i + minJump, farthest + 1);
            int end = min(i + maxJump + 1, n);

            for (int j = start; j < end; ++j) {
                if (j >= n) break;
                if (s[j] == '0') {
                    if (j == n - 1)
                        return true;
                    q.push(j);
                }
            }

            farthest = max(farthest, i + maxJump);
        }

        return false;
    }
};

// Test Cases
int main() {
    Solution sol;

    string s1 = "011010";
    int minJump = 2;
    int maxJump = 3;
    cout << boolalpha << sol.canReach(s1, minJump, maxJump) << endl; // Output: true

    string s2 = "011011101";
    cout << boolalpha << sol.canReach(s2, minJump, maxJump) << endl; // Output: true

    string s3 = "01";
    cout << boolalpha << sol.canReach(s3, 1, 1) << endl; // Output: false

    string s4 = "0";
    cout << boolalpha << sol.canReach(s4, 1, 1) << endl; // Output: true

    return 0;
}

// g++ -std=c++17 Leetcode_1871_Jump_Game_VII.cpp -o test 