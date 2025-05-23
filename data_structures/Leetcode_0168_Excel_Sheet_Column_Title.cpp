#include <iostream>
#include <string>
using namespace std;

class Solution {
public:
    string convertToTitle(int columnNumber) {
        string res = "";
        while (columnNumber > 0) {
            columnNumber--;  // Decrease by 1 because Excel columns are 1-indexed
            int remainder = columnNumber % 26;
            res += (char)('A' + remainder);  // Convert the remainder to a character
            columnNumber /= 26;  // Move to the next significant "digit"
        }

        reverse(res.begin(), res.end());  // Reverse the string to get the correct column name
        return res;
    }
};

int main() {
    Solution solution;
    int columnNumber = 701;  // Example input
    cout << "Column Title: " << solution.convertToTitle(columnNumber) << endl;  // Expected output: "ZY"

    return 0;
}
// g++ Leetcode_0168_Excel_Sheet_Column_Title.cpp -o test  && ./test