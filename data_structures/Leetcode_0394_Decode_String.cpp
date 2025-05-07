#include <iostream>
#include <stack>
#include <string>
using namespace std;

class Solution {
public:
    string decodeString(string s) {
        stack<char> stack;

        for (char& c : s) {
            if (c != ']') {
                stack.push(c);  // Push the current character onto the stack
            } else {
                string sub_str = "";
                
                // Pop characters from stack until we reach '['
                while (stack.top() != '[') {
                    sub_str = stack.top() + sub_str;
                    stack.pop();
                }
                stack.pop();  // Pop the '[' character

                string multiplier = "";
                
                // Pop digits from stack to form the multiplier
                while (!stack.empty() && isdigit(stack.top())) {
                    multiplier = stack.top() + multiplier;
                    stack.pop();
                }

                int repeat_count = stoi(multiplier);  // Convert the multiplier to an integer
                for (int i = 0; i < repeat_count; i++) {
                    for (char& c : sub_str) {
                        stack.push(c);  // Push the decoded substring `repeat_count` times
                    }
                }
            }
        }

        // Convert the stack to the final result string
        string result = "";
        while (!stack.empty()) {
            result = stack.top() + result;
            stack.pop();
        }

        return result;
    }
};

int main() {
    Solution solution;
    string s = "3[a]2[bc]";
    
    string result = solution.decodeString(s);
    cout << "Decoded string: " << result << endl;  // Output should be "aaabcbc"

    return 0;
}

// g++ -std=c++17 Leetcode_0394_Decode_String.cpp -o test