#include <iostream>
#include <vector>
#include <algorithm>  // for sort
using namespace std;

class Solution {
public:
    int hIndex(vector<int>& citations) {
        int n = citations.size();
        sort(citations.begin(), citations.end());  // Sort the citations in ascending order
        int l = 0, r = n - 1;
        
        // Binary search for h-index
        while (l <= r) {
            int mid = (l + r) / 2;

            if (n - mid > citations[mid]) {
                l = mid + 1;
            } else if (n - mid < citations[mid]) {
                r = mid - 1;
            } else {
                return n - mid;
            }
        }

        return n - l;
    }
};

int main() {
    Solution solution;

    // Test case
    vector<int> citations = {3, 0, 6, 1, 5};
    int result = solution.hIndex(citations);

    // Print the result
    cout << "H-Index: " << result << endl;  // Expected output: 3

    return 0;
}

// g++ -std=c++17 Leetcode_0274_H_Index.cpp -o test