#include <iostream>
#include <cmath>
#include <limits>

using namespace std;

class Solution {
public:
    int reverse(int x) {
        int MIN = std::numeric_limits<int>::min();  // -2^31
        int MAX = std::numeric_limits<int>::max();  //  2^31 - 1

        int res = 0;
        while (x != 0) {
            int digit = x % 10;  // Get the last digit of x
            x /= 10;  // Remove the last digit from x

            // Check for overflow or underflow before updating res
            if (res > MAX / 10 || (res == MAX / 10 && digit > MAX % 10)) {
                return 0;  // Overflow
            }
            if (res < MIN / 10 || (res == MIN / 10 && digit < MIN % 10)) {
                return 0;  // Underflow
            }

            // Update res by adding the digit
            res = res * 10 + digit;
        }

        return res;
    }
};

int main() {
    Solution sol;
    int result = sol.reverse(123);
    cout << result << endl;  // Expected Output: 321
    
    return 0;
}


// g++ -std=c++17 Leetcode_0007_Reverse_Integer.cpp -o test