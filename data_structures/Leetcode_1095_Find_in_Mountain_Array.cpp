#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// Mock MountainArray class for testing
class MountainArray {
public:
    vector<int> arr;

    MountainArray(const vector<int>& arr) {
        this->arr = arr;
    }

    int get(int index) {
        return arr[index];
    }

    int length() {
        return arr.size();
    }
};

class Solution {
public:
    // Find the peak index in the mountain array
    int find_peak(int n, MountainArray& mountain_arr) {
        int left = 0, right = n - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (mountain_arr.get(mid) < mountain_arr.get(mid + 1)) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;  // Peak index
    }

    // Binary search function to search within the array
    int binary_search(int left, int right, int target, MountainArray& mountain_arr, bool increasing) {
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int val = mountain_arr.get(mid);
            if (val == target) {
                return mid;  // Found the target
            }

            if (increasing) {
                if (val < target) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            } else {  // Decreasing order
                if (val > target) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return -1;  // Target not found
    }

    // Main function to find the target in the mountain array
    int findInMountainArray(int target, MountainArray& mountain_arr) {
        int n = mountain_arr.length();
        
        // Step 1: Find peak index
        int peak = find_peak(n, mountain_arr);

        // Step 2: Binary search on the increasing part
        int left_result = binary_search(0, peak, target, mountain_arr, true);
        if (left_result != -1) {
            return left_result;  // Found in left part
        }

        // Step 3: Binary search on the decreasing part
        return binary_search(peak + 1, n - 1, target, mountain_arr, false);
    }
};

int main() {
    Solution solution;
    MountainArray mountainArr({1, 2, 3, 4, 5, 3, 1});
    int target = 3;
    
    int result = solution.findInMountainArray(target, mountainArr);
    cout << "Target found at index: " << result << endl;  // Expected output: 2

    return 0;
}



// g++ -std=c++17 Leetcode_1095_Find_in_Mountain_Array.cpp -o test
