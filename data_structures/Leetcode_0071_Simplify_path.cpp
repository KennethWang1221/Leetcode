#include <iostream>
#include <vector>
#include <sstream>
#include <string>
using namespace std;

class Solution {
public:
    string simplifyPath(string path) {
        vector<string> stack;
        stringstream ss(path);
        string token;

        // Split the path by '/'
        while (getline(ss, token, '/')) {
            if (token == "..") {
                // If "..", pop the last element from the stack, if possible
                if (!stack.empty()) {
                    stack.pop_back();
                }
            } else if (token == "." || token.empty()) {
                // Skip "." or empty strings
                continue;
            } else {
                // Otherwise, push the valid directory name onto the stack
                stack.push_back(token);
            }
        }

        // Build the simplified path
        string result = "/";
        for (int i = 0; i < stack.size(); i++) {
            result += stack[i];
            if (i < stack.size() - 1) {
                result += "/";
            }
        }

        return result;
    }
};

int main() {
    Solution solution;
    string path = "/../abc//./def/";

    string result = solution.simplifyPath(path);
    cout << "Simplified Path: " << result << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0071_Simplify_path.cpp -o test