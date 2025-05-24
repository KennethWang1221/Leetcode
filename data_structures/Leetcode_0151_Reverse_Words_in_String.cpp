#include <iostream>
#include <sstream>
#include <vector>
using namespace std;

class Solution {
public:
    string reverseWords(string s) {
        // Remove leading and trailing spaces
        int n = s.length();
        int left = 0, right = n - 1;
        
        // Skip leading spaces
        while (left < n && s[left] == ' ') {
            left++;
        }

        // Skip trailing spaces
        while (right >= left && s[right] == ' ') {
            right--;
        }

        // Store words in a vector
        vector<string> words;
        string word;
        for (int i = left; i <= right; ++i) {
            if (s[i] == ' ') {
                if (!word.empty()) {
                    words.push_back(word);
                    word = "";
                }
            } else {
                word += s[i];
            }
        }
        // Add the last word
        if (!word.empty()) {
            words.push_back(word);
        }

        // Reverse the words
        string result = "";
        for (int i = words.size() - 1; i >= 0; --i) {
            result += words[i];
            if (i != 0) {
                result += " ";
            }
        }

        return result;
    }
};

int main() {
    Solution solution;

    // Test case
    string s = "the sky is blue";
    // string s = "  hello world  ";
    // string s = "a good   example";
    string result = solution.reverseWords(s);

    // Print the result
    cout << "Reversed Words: \"" << result << "\"" << endl;  // Expected: "blue is sky the"

    return 0;
}

// g++ -std=c++17 Leetcode_0151_Reverse_Words_in_String.cpp -o test
