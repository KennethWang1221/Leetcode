#include <iostream>
using namespace std;

class Solution {
public:
    int tribonacci(int n) {
        if (n == 0) return 0;
        if (n == 1 || n == 2) return 1;

        int t0 = 0, t1 = 1, t2 = 1;

        for (int i = 3; i <= n; ++i) {
            int next = t0 + t1 + t2;
            t0 = t1;
            t1 = t2;
            t2 = next;
        }

        return t2;
    }
};

// Test cases
int main() {
    Solution sol;
    cout << "tribonacci(0): " << sol.tribonacci(0) << endl; // Output: 0
    cout << "tribonacci(1): " << sol.tribonacci(1) << endl; // Output: 1
    cout << "tribonacci(2): " << sol.tribonacci(2) << endl; // Output: 1
    cout << "tribonacci(4): " << sol.tribonacci(4) << endl; // Output: 4
    cout << "tribonacci(25): " << sol.tribonacci(25) << endl; // Output: 1389537
    return 0;
}

// g++ -std=c++17 Leetcode_1137_N-th_Tribonacci_Number.cpp -o test