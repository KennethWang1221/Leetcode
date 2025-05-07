#include <iostream>
#include <vector>
#include <string>
using namespace std;

class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> res;
        string stack;

        // Backtracking function to generate the parentheses
        backtrack(n, 0, 0, stack, res);
        return res;
    }

private:
    void backtrack(int n, int openN, int closedN, string& stack, vector<string>& res) {
        // If the number of open and closed parentheses is equal to n, add the current stack as a result
        if (openN == closedN && openN == n) {
            res.push_back(stack);
            return;
        }

        // If there are still open parentheses left, add an open parenthesis and recurse
        if (openN < n) {
            stack.push_back('(');
            backtrack(n, openN + 1, closedN, stack, res);
            stack.pop_back();  // Backtrack
        }

        // If there are more open parentheses than closed ones, add a closed parenthesis and recurse
        if (closedN < openN) {
            stack.push_back(')');
            backtrack(n, openN, closedN + 1, stack, res);
            stack.pop_back();  // Backtrack
        }
    }
};

int main() {
    Solution solution;
    int n = 3;
    
    vector<string> result = solution.generateParenthesis(n);
    
    cout << "Generated parentheses for n=" << n << ":" << endl;
    for (const string& str : result) {
        cout << str << endl;
    }

    return 0;
}

// g++ -std=c++17 Leetcode_0022_Generate_Parentheses.cpp -o test