#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    string intToRoman(int num) {
        vector<pair<string, int>> symList = {
            {"M", 1000}, {"CM", 900}, {"D", 500}, {"CD", 400}, {"C", 100}, {"XC", 90}, 
            {"L", 50}, {"XL", 40}, {"X", 10}, {"IX", 9}, {"V", 5}, {"IV", 4}, {"I", 1}
        };

        string res = "";

        for (auto& [sym, val] : symList) {
            while (num >= val) {
                res += sym;
                num -= val;
            }
        }

        return res;
    }
};

int main() {
    Solution solution;

    // Test case
    int num = 1994;
    string result = solution.intToRoman(num);

    // Print the result
    cout << "Roman numeral: " << result << endl;  // Expected output: "MCMXCIV"

    return 0;
}

// g++ -std=c++17 Leetcode_0012_Integer_to_Roman.cpp -o test