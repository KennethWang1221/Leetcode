#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maxTurbulenceSize(vector<int>& arr) {
        int n = arr.size();
        if (n <= 1)
            return n;

        int maxLen = 1;
        int up = 1, down = 1;

        for (int i = 1; i < n; ++i) {
            if (arr[i] > arr[i - 1]) {
                up = down + 1;
                down = 1;
            } else if (arr[i] < arr[i - 1]) {
                down = up + 1;
                up = 1;
            } else {
                // Equal elements: reset both
                up = 1;
                down = 1;
            }
            maxLen = max(maxLen, max(up, down));
        }

        return maxLen;
    }
};

int main() {
    Solution solution;
    
    // Example usage
    vector<int> arr = {9, 4, 2, 10, 7, 8, 8, 1, 9};
    cout << "Max turbulence size: " << solution.maxTurbulenceSize(arr) << endl;  // Expected output: 5

    return 0;
}

// g++ -std=c++17 Leetcode_0978_Longest_Turbulent_Subarray.cpp -o test 