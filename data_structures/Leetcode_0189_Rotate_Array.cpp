#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    vector<int> rotate(vector<int>& nums, int k) {
        int n = nums.size();
        k = k % n;  // Handle cases where k is larger than n
        int l = 0, r = n - 1;

        // Reverse the entire array
        while (l < r) {
            swap(nums[l], nums[r]);
            l++;
            r--;
        }

        // Reverse the first k elements
        l = 0, r = k - 1;
        while (l < r) {
            swap(nums[l], nums[r]);
            l++;
            r--;
        }

        // Reverse the remaining n - k elements
        l = k, r = n - 1;
        while (l < r) {
            swap(nums[l], nums[r]);
            l++;
            r--;
        }

        return nums;
    }
};

int main() {
    Solution solution;
    vector<int> nums = {1, 2, 3, 4, 5, 6, 7};
    int k = 3;

    vector<int> result = solution.rotate(nums, k);

    // Output the rotated array
    cout << "Rotated array: ";
    for (int num : result) {
        cout << num << " ";
    }
    cout << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0189_Rotate_Array.cpp -o test
