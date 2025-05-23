#include <iostream>
#include <vector>
#include <string>
using namespace std;

class Solution {
public:
    string multiply(string num1, string num2) {
        if (num1 == "0" || num2 == "0") {
            return "0";
        }

        int m = num1.size();
        int n = num2.size();
        vector<int> res(m + n, 0);

        // Reverse both strings to easily handle digit-wise multiplication
        reverse(num1.begin(), num1.end());
        reverse(num2.begin(), num2.end());

        // Perform the multiplication
        for (int i1 = 0; i1 < m; ++i1) {
            for (int i2 = 0; i2 < n; ++i2) {
                int digit = (num1[i1] - '0') * (num2[i2] - '0');
                res[i1 + i2] += digit;
                res[i1 + i2 + 1] += res[i1 + i2] / 10;
                res[i1 + i2] = res[i1 + i2] % 10;
            }
        }

        // Remove leading zeros from the result
        int beg = res.size() - 1;
        while (beg >= 0 && res[beg] == 0) {
            --beg;
        }

        // Convert the result vector to string
        string result = "";
        for (int i = beg; i >= 0; --i) {
            result += (res[i] + '0');
        }

        return result;
    }
};

int main() {
    Solution solution;

    // Test case
    string num1 = "2", num2 = "3";
    string result = solution.multiply(num1, num2);

    // Print the result
    cout << "Result: " << result << endl;  // Expected output: "6"

    return 0;
}
// g++ -std=c++17 Leetcode_0043_Multiply_Strings.cpp -o test