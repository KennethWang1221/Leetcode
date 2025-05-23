#include <iostream>
#include <string>

using namespace std;

class Solution {
public:
    string gcdOfStrings(string str1, string str2) {
        int len1 = str1.size();
        int len2 = str2.size();

        // Helper lambda to check if a substring of length `l` divides both strings
        auto isDivisor = [&](int l) -> bool {
            if (len1 % l != 0 || len2 % l != 0)
                return false;
            int f1 = len1 / l, f2 = len2 / l;

            string base = str1.substr(0, l);

            string s1 = "", s2 = "";
            for (int i = 0; i < f1; ++i) s1 += base;
            for (int i = 0; i < f2; ++i) s2 += base;

            return s1 == str1 && s2 == str2;
        };

        int maxLen = min(len1, len2);
        for (int l = maxLen; l > 0; --l) {
            if (isDivisor(l)) {
                return str1.substr(0, l);
            }
        }

        return "";
    }
};

// Test Case
int main() {
    Solution sol;
    string result = sol.gcdOfStrings("ABABAB", "ABAB");
    cout << "GCD String: " << result << endl; // Output: AB

    // Additional test cases
    cout << sol.gcdOfStrings("ABCABC", "ABC") << endl; // Output: ABC
    cout << sol.gcdOfStrings("ABABAB", "ABAC") << endl; // Output: ""
    return 0;
}
// g++ Leetcode_1071_Greatest_Common_Divisor_of_String.cpp -o test  && ./test