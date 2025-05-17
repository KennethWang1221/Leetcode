#include <iostream>
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.size();
        if (n <= 1) return s;  // If the string has length 1 or less, return the string itself
        
        int Max_Len = 1;
        string Max_Str = s.substr(0, 1);  // Initialize the result with the first character
        
        // Iterate through each possible substring
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (j - i + 1 > Max_Len && isPalindrome(s, i, j)) {
                    Max_Len = j - i + 1;
                    Max_Str = s.substr(i, j - i + 1);
                }
            }
        }
        
        return Max_Str;
    }

private:
    bool isPalindrome(const string& s, int start, int end) {
        // Check if the substring s[start...end] is a palindrome
        while (start < end) {
            if (s[start] != s[end]) {
                return false;
            }
            ++start;
            --end;
        }
        return true;
    }
};

int main() {
    Solution sol;
    string s = "babad";
    string result = sol.longestPalindrome(s);
    cout << result << endl;  // Expected output: "bab" or "aba"
    
    return 0;
}


// g++ -std=c++17 Leetcode_0005_Longest_Palindromic_Substring.cpp -o test