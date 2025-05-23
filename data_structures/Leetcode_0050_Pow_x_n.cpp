#include <iostream>
#include <cmath>
using namespace std;

class Solution {
public:
    // Helper function for recursive power calculation
    double helper(double x, long n) {
        if (n == 0) return 1;
        double res = helper(x, n / 2);
        res *= res;
        return (n % 2 == 0) ? res : x * res;
    }

    double myPow(double x, int n) {
        if (x == 0) return 0;
        long exp = n; // Use long to handle edge case of INT_MIN
        double result = helper(x, abs(exp));
        return (exp >= 0) ? result : 1 / result;
    }
};

int main() {
    Solution solution;
    
    // Test case
    double x = 2.00000;
    int n = 10;
    double result = solution.myPow(x, n);
    
    cout << "Result: " << result << endl;  // Expected output: 1024
    
    return 0;
}

// g++ -std=c++17 Leetcode_0050_Pow_x_n.cpp -o test