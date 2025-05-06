#include <iostream>
#include <vector>
using namespace std;

// bubble sort version
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        // Copy nums2 into the end of nums1
        for (int i = 0; i < n; i++) {
            nums1[m + i] = nums2[i];
        }

        // Perform a simple bubble sort
        int size = m + n;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size - 1 - i; j++) {
                if (nums1[j] > nums1[j + 1]) {
                    swap(nums1[j], nums1[j + 1]);
                }
            }
        }

        // Output the merged and sorted nums1
        for (int i = 0; i < size; i++) {
            cout << nums1[i] << " ";
        }
        cout << endl;
    }



    void merge_two_pointers(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        // Merge the arrays in-place from the end of nums1
        while (m > 0 && n > 0) {
            if (nums1[m - 1] >= nums2[n - 1]) {
                nums1[m + n - 1] = nums1[m - 1];
                m--;
            } else {
                nums1[m + n - 1] = nums2[n - 1];
                n--;
            }
        }

        // If there are still elements in nums2, copy them
        if (n > 0) {
            for (int i = 0; i < n; i++) {
                nums1[i] = nums2[i];
            }
        }

        // Output the merged nums1
        for (int i = 0; i < m + n; i++) {
            cout << nums1[i] << " ";
        }
        cout << endl;
    }
};

int main() {
    Solution solution;
    vector<int> nums1 = {2, 0};  // You can test with this input
    int m = 1;
    vector<int> nums2 = {1};
    int n = 1;
    solution.merge(nums1, m, nums2, n);
    solution.merge_two_pointers(nums1, m, nums2, n);

    // Test with the second example
    vector<int> nums1_2 = {1, 2, 3, 0, 0, 0};
    m = 3;
    vector<int> nums2_2 = {2, 5, 6};
    n = 3;
    solution.merge(nums1_2, m, nums2_2, n);
    solution.merge_two_pointers(nums1_2, m, nums2_2, n);

    return 0;
}
// g++ -std=c++17 Leetcode_0088_Merge_Sorted_Array.cpp -o test
