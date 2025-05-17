#include <iostream>

using namespace std;

class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        uint32_t res = 0;
        for (int i = 0; i < 32; ++i) {
            // Get the i-th bit of n
            uint32_t bit = (n >> i) & 1;
            // Set the (31 - i)-th bit of res
            res |= (bit << (31 - i));
        }
        return res;
    }
};

int main() {
    Solution sol;
    uint32_t n = 0b00000010100101000001111010011100;  // Input number
    uint32_t result = sol.reverseBits(n);
    cout << result << endl;  // Expected Output: 964176192 (in decimal)
    
    return 0;
}
// g++ -std=c++17 Leetcode_0190_Reverse_Bits.cpp -o test