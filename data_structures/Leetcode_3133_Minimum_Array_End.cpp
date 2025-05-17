#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    long long minEnd(long long n, long long x) {
        long long cnt = n - 1;
        long long result = x;
        long long bit = 1;

        // Apply the bits of 'cnt' into the zero positions of 'x'
        for (int i = 0; cnt > 0; ++i) {
            if ((result >> i & 1) == 0) {
                if (cnt & 1)
                    result |= (1LL << i);
                cnt >>= 1;
            }
        }

        return result;
    }
};

int main() {
    Solution sol;
    int n = 6715154;
    int x = 7193485;
    cout << sol.minEnd(n, x) << endl;
    
    return 0;
}


// g++ -std=c++17 Leetcode_3133_Minimum_Array_End.cpp -o test