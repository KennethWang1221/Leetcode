#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

class Solution {
public:
    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        int l = 0, r = arr.size() - 1;
        
        // Find the index of x or the closest value to x using binary search
        int val = arr[0], idx = 0;
        while (l <= r) {
            int m = (l + r) / 2;
            int curDiff = abs(arr[m] - x);
            int resDiff = abs(val - x);

            // Update the closest value and its index
            if (curDiff < resDiff || (curDiff == resDiff && arr[m] < val)) {
                val = arr[m];
                idx = m;
            }

            // Binary search logic
            if (arr[m] < x) {
                l = m + 1;
            } else if (arr[m] > x) {
                r = m - 1;
            } else {
                break;
            }
        }

        l = r = idx;
        
        // Expand the window to find k closest elements
        for (int i = 0; i < k - 1; i++) {
            if (l == 0) {
                r++;
            } else if (r == arr.size() - 1 || x - arr[l - 1] <= arr[r + 1] - x) {
                l--;
            } else {
                r++;
            }
        }

        // Create the result vector with k closest elements
        vector<int> result;
        for (int i = l; i <= r; i++) {
            result.push_back(arr[i]);
        }

        return result;
    }
};

int main() {
    Solution solution;
    vector<int> arr = {1, 2, 3, 4, 5};
    int k = 4;
    int x = 3;

    vector<int> result = solution.findClosestElements(arr, k, x);
    
    cout << "The " << k << " closest elements to " << x << " are: ";
    for (int num : result) {
        cout << num << " ";
    }
    cout << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0658_Find_K_Closes_Elements.cpp -o test