#include <iostream>
#include <string>
using namespace std;

class Solution {
public:
    string convert(string s, int numRows) {
        if (numRows <= 1) return s;

        int n = s.length();
        int increment = (numRows - 1) * 2;
        string result = "";

        for (int r = 0; r < numRows; ++r) {
            for (int i = r; i < n; i += increment) {
                result += s[i];

                int middle = i + increment - 2 * r;
                if (r > 0 && r < numRows - 1 && middle < n) {
                    result += s[middle];
                }
            }
        }

        return result;
    }
};

int main() {
    Solution solution;

    // Test case
    string s = "PAYPALISHIRING";
    int numRows = 4;
    string result = solution.convert(s, numRows);

    // Print the result
    cout << "Zigzag pattern: " << result << endl;  // Expected output: "PINALSIGYAHRPI"

    return 0;
}

// g++ -std=c++17 Leetcode_0006_ZigZag_Conversion.cpp -o test