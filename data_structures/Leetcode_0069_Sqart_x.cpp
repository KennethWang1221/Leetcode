#include <iostream>
using namespace std;

class Solution {
public:
    int mySqrt(int x) {
        long long l = 0, r = x;

        while (l <= r) {
            long long mid = l + (r - l) / 2;
            if (mid * mid == x) {
                return mid;
            }
            if (mid * mid < x) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }

        return r;
    }
};

int main() {
    Solution solution;
    int x = 8;
    
    int result = solution.mySqrt(x);
    cout << "The integer square root of " << x << " is: " << result << endl;  // Expected output: 2

    return 0;
}



// g++ -std=c++17 Leetcode_0069_Sqart_x.cpp -o test