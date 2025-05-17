#include <iostream>
#include <cmath>

using namespace std;

class Solution {
public:
    int add(int a, int b) {
        if (a == 0 || b == 0) {
            return a | b;  // If either number is zero, return the other number
        }
        return add((a ^ b), ((a & b) << 1));  // Add using XOR and carry (AND + shift)
    }

    int getSum(int a, int b) {
        // If a * b < 0, one is positive, the other is negative
        if (a * b < 0) {
            if (a > 0) {
                return getSum(b, a);  // Swap if a is positive
            }
            if (add(~a, 1) == b) {  // If -a == b, return 0
                return 0;
            }
            if (add(~a, 1) < b) {  // If -a < b, use recursive addition for negative numbers
                return add(~add(add(~a, 1), add(~b, 1)), 1);
            }
        }

        // If both a and b are non-negative or both are negative, simply use the add function
        return add(a, b);
    }
};

int main() {
    Solution sol;
    int result = sol.getSum(1, 2);
    cout << result << endl;  // Expected output: 3
    
    return 0;
}
// g++ -std=c++17 Leetcode_0371_Sum_of_Two_Integers.cpp -o test