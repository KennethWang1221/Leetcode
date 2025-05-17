#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> countBits(int n) {
        vector<int> res(n + 1, 0);  // Initialize a vector with size n+1 and all values as 0
        for (int i = 1; i <= n; ++i) {
            if (i % 2 == 1) {
                res[i] = res[i - 1] + 1;  // If odd, the count of 1's is one more than the previous number
            } else {
                res[i] = res[i / 2];  // If even, it has the same number of 1's as i/2
            }
        }
        return res;
    }
};

int main() {
    Solution sol;
    int n = 2;
    vector<int> result = sol.countBits(n);
    
    // Print the result
    for (int i : result) {
        cout << i << " ";
    }
    cout << endl;  // Expected Output: [0, 1, 1]

    return 0;
}
// g++ -std=c++17 Leetcode_0338_Counting_Bits.cpp -o test