#include <iostream>
#include <vector>
#include <stack>
#include <string>
#include <cstdlib>  // For stoi (string to integer conversion)
using namespace std;

class Solution {
public:
    int evalRPN(vector<string>& tokens) {
        stack<int> stack;
        
        for (const string& t : tokens) {
            if (t == "+") {
                int a1 = stack.top(); stack.pop();
                int a2 = stack.top(); stack.pop();
                stack.push(a2 + a1);
            } else if (t == "-") {
                int a1 = stack.top(); stack.pop();
                int a2 = stack.top(); stack.pop();
                stack.push(a2 - a1);
            } else if (t == "*") {
                int a1 = stack.top(); stack.pop();
                int a2 = stack.top(); stack.pop();
                stack.push(a2 * a1);
            } else if (t == "/") {
                int a1 = stack.top(); stack.pop();
                int a2 = stack.top(); stack.pop();
                stack.push(int(a2 / a1));  // Casting to int to match Python's behavior
            } else {
                stack.push(stoi(t));  // Convert string to integer and push onto stack
            }
        }
        
        return stack.top();  // Return the result which is the last element in the stack
    }
};

int main() {
    Solution solution;

    vector<string> tokens1 = {"2", "1", "+", "3", "*"};
    vector<string> tokens2 = {"4", "13", "5", "/", "+"};
    vector<string> tokens3 = {"10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"};
    
    cout << "Result 1: " << solution.evalRPN(tokens1) << endl;  // Output: 9
    cout << "Result 2: " << solution.evalRPN(tokens2) << endl;  // Output: 6
    cout << "Result 3: " << solution.evalRPN(tokens3) << endl;  // Output: 22

    return 0;
}

// g++ -std=c++17 Leetcode_0150_evalRPN.cpp -o test