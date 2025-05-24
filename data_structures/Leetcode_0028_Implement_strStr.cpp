#include <iostream>
#include <string>
using namespace std;

class Solution {
public:
    int strStr(string haystack, string needle) {
        // Use the find method of string to find the first occurrence of needle in haystack
        size_t pos = haystack.find(needle);
        
        // If needle is found, return its index; otherwise, return -1
        return (pos != string::npos) ? pos : -1;
    }
};

int main() {
    Solution solution;
    
    // Test case
    string haystack = "hello";
    string needle = "ll";
    int result = solution.strStr(haystack, needle);

    // Print the result
    cout << "Result: " << result << endl;  // Expected output: 2

    return 0;
}

// g++ -std=c++17 Leetcode_0028_Implement_strStr.cpp -o test
