#include <iostream>

using namespace std;

class Solution {
public:
    int hammingWeight(uint32_t n) {
        int res = 0;
        while (n) {
            n &= (n - 1);  // Drop the lowest set bit
            res += 1;  // Increment the count of 1s
        }
        return res;
    }
};

int main() {
    Solution sol;
    uint32_t n = 11;  // Binary: 1011
    int result = sol.hammingWeight(n);
    cout << result << endl;  // Output: 3 (because 11 in binary is 1011, which has 3 ones)

    return 0;
}


// g++ -std=c++17 Leetcode_0191_Number_of_1_Bits.cpp -o test