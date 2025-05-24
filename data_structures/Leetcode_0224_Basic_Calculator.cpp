#include <iostream>
#include <vector>
#include <cctype>

using namespace std;

class Solution {
public:
    int calculate(string s) {
        int num = 0;
        int sign = 1;
        int res = 0;
        vector<int> stack;

        for (int i = 0; i < s.size(); ++i) {
            char c = s[i];

            if (isdigit(c)) {
                num = num * 10 + (c - '0');
            } else if (c == '+') {
                res += sign * num;
                num = 0;
                sign = 1;
            } else if (c == '-') {
                res += sign * num;
                num = 0;
                sign = -1;
            } else if (c == '(') {
                // Push current result and sign onto stack
                stack.push_back(res);
                stack.push_back(sign);
                res = 0;
                sign = 1;
            } else if (c == ')') {
                res += sign * num;
                res *= stack.back(); stack.pop_back(); // Apply sign before parenthesis
                res += stack.back(); stack.pop_back(); // Add previous result
                num = 0;
            }
        }

        return res + sign * num;
    }
};

// Test Case
int main() {
    Solution sol;
    string s1 = "(1+(4+5+2)-3)+(6+8)";
    cout << sol.calculate(s1) << endl; // Output: 25

    string s2 = "1+2";
    cout << sol.calculate(s2) << endl; // Output: 3

    string s3 = "(2+3)-(4-5)";
    cout << sol.calculate(s3) << endl; // Output: 6

    return 0;
}

// g++ -std=c++17 Leetcode_0224_Basic_Calculator.cpp -o test