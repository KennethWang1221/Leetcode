#include <iostream>

using namespace std;

class Solution {
public:
    int rangeBitwiseAnd(int left, int right) {
        int i = 0;
        while (left != right) {
            left >>= 1;  // Right shift left by 1 bit
            right >>= 1;  // Right shift right by 1 bit
            i++;  // Keep track of the number of shifts
        }
        return left << i;  // Left shift the result back to get the correct answer
    }
};

int main() {
    Solution sol;
    int left = 5;
    int right = 7;
    int result = sol.rangeBitwiseAnd(left, right);
    cout << result << endl;  // Expected output: 4
    
    return 0;
}


// g++ -std=c++17 Leetcode_0201_Bitwise_AND_of_Numbers_Range.cpp -o test