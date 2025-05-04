#include <iostream>
#include <vector>

using namespace std;

// Merge function to merge two sorted halves
vector<int> merge(const vector<int>& left, const vector<int>& right) {
    vector<int> result;
    int i = 0, j = 0;
    
    // Merge the two halves while both have elements
    while (i < left.size() && j < right.size()) {
        if (left[i] < right[j]) {
            result.push_back(left[i]);
            i++;
        } else {
            result.push_back(right[j]);
            j++;
        }
    }

    // If any elements remain in left, add them
    result.insert(result.end(), left.begin() + i, left.end());

    // If any elements remain in right, add them
    result.insert(result.end(), right.begin() + j, right.end());

    return result;
}

// Merge sort function that divides and sorts the array
vector<int> merge_sort(const vector<int>& array) {
    int n = array.size();
    if (n <= 1) {
        return array;
    }

    int middle = n / 2;
    vector<int> left(array.begin(), array.begin() + middle);
    vector<int> right(array.begin() + middle, array.end());

    left = merge_sort(left);
    right = merge_sort(right);

    return merge(left, right);
}

class Solution {
public:
    vector<int> sortArray(vector<int>& nums) {
        return merge_sort(nums);  // Call merge_sort to sort the array
    }
};

int main() {
    Solution solution;
    vector<int> nums = {2, 4, 9, 1, 7, 8, 3, 14, 19, 16};
    
    vector<int> sorted_array = solution.sortArray(nums);

    // Print sorted array
    for (int num : sorted_array) {
        cout << num << " ";
    }
    cout << endl;

    return 0;
}
// g++ -std=c++17 Leetcode_0912_Sort_an_Array.cpp -o test
