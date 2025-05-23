#include <iostream>
#include <unordered_map>
using namespace std;

class Solution {
public:
    int romanToInt(string s) {
        unordered_map<char, int> roman = {
            {'I', 1}, {'V', 5}, {'X', 10}, {'L', 50}, {'C', 100}, {'D', 500}, {'M', 1000}
        };
        
        int res = 0;
        int n = s.size();
        
        for (int i = 0; i < n; ++i) {
            // If the current value is less than the next value, subtract it
            if (i + 1 < n && roman[s[i]] < roman[s[i + 1]]) {
                res -= roman[s[i]];
            } else {
                res += roman[s[i]];
            }
        }
        
        return res;
    }
};

int main() {
    Solution solution;

    // Test case
    string s = "MCMXCIV";
    int result = solution.romanToInt(s);

    // Print the result
    cout << "Roman to Integer: " << result << endl;  // Expected output: 1994

    return 0;
}

// g++ -std=c++17 Leetcode_0013_Roman_to_Integer.cpp -o test
// ./test