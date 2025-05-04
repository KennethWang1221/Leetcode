#include <iostream>
#include <vector>
#include <string>

class Solution {
public:
    std::string longestCommonPrefix(std::vector<std::string>& strs) {
        std::string s1 = strs[0];
        int n = s1.length();
        std::string res = "";
        
        for (int i = 0; i < n; ++i) {
            char substring = s1[i];
            for (const std::string& s : strs) {
                if (i >= s.length() || s[i] != substring) {
                    return res; // Prefix no longer matches
                }
            }
            res += substring; // Add the matching character to result
        }
        
        return res;
    }
};

int main() {
    Solution solution;
    std::vector<std::string> strs = {"fl", "flower", "flight"};
    std::string result = solution.longestCommonPrefix(strs);
    
    std::cout << result << std::endl; // Output will be "fl"
    
    return 0;
}
// g++ -std=c++17 Leetcode_0014_Longest_Common_Prefix.cpp -o test