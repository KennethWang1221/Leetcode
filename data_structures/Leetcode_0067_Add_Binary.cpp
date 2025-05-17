#include <iostream>
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
    string addBinary(string a, string b) {
        string res = "";
        int carry = 0;
        int i = a.size() - 1, j = b.size() - 1;
        
        while (i >= 0 || j >= 0 || carry) {
            int digitA = (i >= 0) ? a[i] - '0' : 0;
            int digitB = (j >= 0) ? b[j] - '0' : 0;
            
            int total = digitA + digitB + carry;
            res = to_string(total % 2) + res;
            carry = total / 2;
            
            i--;
            j--;
        }
        
        return res;
    }
};

int main() {
    Solution sol;
    string a = "1010";
    string b = "1011";
    
    string result = sol.addBinary(a, b);
    cout << result << endl;  // Expected Output: "10101"
    
    return 0;
}
// g++ -std=c++17 Leetcode_0067_Add_Binary.cpp -o test