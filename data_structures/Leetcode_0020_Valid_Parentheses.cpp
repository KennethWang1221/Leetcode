#include <iostream>
#include <stack>
#include <unordered_map>
using namespace std;

class Solution {
public:
    bool isValid(string s) {
        unordered_map<char, char> preMap = {{'(', ')'}, {'{', '}'}, {'[', ']'}};
        stack<char> candidates;

        for (char c : s) {
            if (preMap.find(c) != preMap.end()) {
                // If the character is one of the opening brackets, push the corresponding closing bracket
                candidates.push(preMap[c]);
            } else {
                // If the character is a closing bracket, check if it matches the top of the stack
                if (candidates.empty() || candidates.top() != c) {
                    return false;
                }
                candidates.pop();  // Pop the matched opening bracket
            }
        }

        return candidates.empty();  // If the stack is empty, all brackets are matched
    }
};

int main() {
    Solution solution;
    
    string s1 = "({{{{}}}))";
    cout << (solution.isValid(s1) ? "Valid" : "Invalid") << endl;
    
    string s2 = "]";
    cout << (solution.isValid(s2) ? "Valid" : "Invalid") << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_0020_Valid_Parentheses.cpp -o test