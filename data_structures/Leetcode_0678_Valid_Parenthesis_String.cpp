#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;

class Solution {
    string s;
    int n;
    unordered_map<int, bool> dp;

    bool dfs(int i, int left) {
        if (i == n || left < 0) {
            return left == 0;
        }

        int key = i * 10000 + left;
        if (dp.count(key)) return dp[key];

        if (s[i] == '(') {
            return dp[key] = dfs(i + 1, left + 1);
        } else if (s[i] == ')') {
            return dp[key] = dfs(i + 1, left - 1);
        } else {
            // '*' can be '(', ')', or ''
            return dp[key] = dfs(i + 1, left + 1) || dfs(i + 1, left - 1) || dfs(i + 1, left);
        }
    }

public:
    // Greedy O(n) solution
    bool checkValidString_greedy(string input) {
        int leftMin = 0, leftMax = 0;

        for (char c : input) {
            if (c == '(') {
                leftMin++;
                leftMax++;
            } else if (c == ')') {
                leftMin--;
                leftMax--;
            } else { // '*'
                leftMin--;   // '*' treated as ')'
                leftMax++;   // '*' treated as '('
            }

            if (leftMax < 0)
                return false;

            if (leftMin < 0)
                leftMin = 0;  // We can't have negative open count
        }

        return leftMin == 0;
    }

    // DP O(n^2) solution
    bool checkValidString_dp(string input) {
        s = input;
        n = s.size();
        dp.clear();
        return dfs(0, 0);
    }
};

// Test Case
int main() {
    Solution sol;

    string s1 = "()";
    cout << "Greedy: " << boolalpha << sol.checkValidString_greedy(s1) << endl; // true

    string s2 = "(*)";
    cout << "DP: " << boolalpha << sol.checkValidString_dp(s2) << endl; // true

    string s3 = "(*))";
    cout << "Greedy: " << boolalpha << sol.checkValidString_greedy(s3) << endl; // true

    string s4 = ")(";
    cout << "DP: " << boolalpha << sol.checkValidString_dp(s4) << endl; // false

    return 0;
}
// g++ -std=c++11 Leetcode_0678_Valid_Parenthesis_String.cpp -o ./test && ./test