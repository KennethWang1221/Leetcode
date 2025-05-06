#include <iostream>
#include <cctype>
#include <string>
using namespace std;

class Solution {
public:
    bool isPalindrome(string s) {
        int l = 0, r = s.length() - 1;

        while (l < r) {
            // Skip non-alphanumeric characters
            while (l < r && !isAlpha(s[l])) {
                l++;
            }
            while (l < r && !isAlpha(s[r])) {
                r--;
            }

            // Compare characters, ignoring case
            if (tolower(s[l]) != tolower(s[r])) {
                return false;
            }

            l++;
            r--;
        }

        return true;
    }

private:
    bool isAlpha(char i) {
        return (isalpha(i) || isdigit(i)); // Check if character is alphanumeric
    }
};

int main() {
    Solution solution;
    string s = "A man, a plan, a canal: Panama"; // You can test with other strings
    bool res = solution.isPalindrome(s);
    cout << (res ? "True" : "False") << endl; // Output the result
    return 0;
}

// g++ -std=c++17 Leetcode_0125_Valid_Palindrome.cpp -o test