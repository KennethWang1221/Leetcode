#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    void sortColors(vector<int>& nums) {
        int p = 0, q = 0, k = nums.size() - 1;

        while (q <= k) {
            if (nums[q] == 0) {
                swap(nums[p], nums[q]);
                p++;
                q++;
            } else if (nums[q] == 2) {
                swap(nums[k], nums[q]);
                k--;
            } else {
                q++;
            }
        }
    }

    void sortColors_bubble_sort(vector<int>& nums) {
        int n = nums.size();
        for (int i = 0; i < n-1; i++){
            for (int j = 0; j < n-i-1; j++){
                if (nums[j] > nums[j+1]){
                    int temp = nums[j];
                    nums[j] = nums[j+1];
                    nums[j+1] = temp;
                }
            }
        }
        
    }
};

int main() {
    Solution solution;

    vector<int> nums1 = {2, 0, 2, 1, 1, 0};
    solution.sortColors(nums1);
    for (int num : nums1) {
        cout << num << " ";  // Expected output: 0 0 1 1 2 2
    }
    cout << endl;

    vector<int> nums2 = {0, 1, 0, 1, 2, 2};
    solution.sortColors(nums2);
    for (int num : nums2) {
        cout << num << " ";  // Expected output: 0 0 1 1 2 2
    }
    cout << endl;

    vector<int> nums3 = {2, 0, 1};
    solution.sortColors_bubble_sort(nums3);
    for (int num : nums3) {
        cout << num << " ";  // Expected output: 0 1 2
    }
    cout << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_0075_Sort_Colors.cpp -o test
