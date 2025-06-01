#include <iostream>
#include <vector>
#include <queue>
#include <tuple>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        if (nums1.empty() || nums2.empty() || k <= 0) {
            return {};
        }
        
        auto cmp = [](const tuple<int, int, int>& a, const tuple<int, int, int>& b) {
            return get<0>(a) > get<0>(b);
        };
        priority_queue<tuple<int, int, int>, vector<tuple<int, int, int>>, decltype(cmp)> min_heap(cmp);
        
        int n1 = nums1.size();
        int n2 = nums2.size();
        for (int i = 0; i < min(n1, k); ++i) {
            min_heap.push(make_tuple(nums1[i] + nums2[0], i, 0));
        }
        
        vector<vector<int>> result;
        while (!min_heap.empty() && result.size() < k) {
            auto [sum, i, j] = min_heap.top();
            min_heap.pop();
            result.push_back({nums1[i], nums2[j]});
            if (j + 1 < n2) {
                min_heap.push(make_tuple(nums1[i] + nums2[j+1], i, j+1));
            }
        }
        
        return result;
    }
};

int main() {
    Solution solution;
    
    vector<int> nums1 = {1, 7, 11};
    vector<int> nums2 = {2, 4, 6};
    int k = 3;
    vector<vector<int>> result = solution.kSmallestPairs(nums1, nums2, k);
    cout << "Test case 1: ";
    for (auto& pair : result) {
        cout << "(" << pair[0] << ", " << pair[1] << ") ";
    }
    cout << endl; // Expected: (1, 2) (1, 4) (1, 6)

    nums1 = {1, 1, 2};
    nums2 = {1, 2, 3};
    k = 2;
    result = solution.kSmallestPairs(nums1, nums2, k);
    cout << "Test case 2: ";
    for (auto& pair : result) {
        cout << "(" << pair[0] << ", " << pair[1] << ") ";
    }
    cout << endl; // Expected: (1, 1) (1, 1)

    nums1 = {1, 2};
    nums2 = {3};
    k = 3;
    result = solution.kSmallestPairs(nums1, nums2, k);
    cout << "Test case 3: ";
    for (auto& pair : result) {
        cout << "(" << pair[0] << ", " << pair[1] << ") ";
    }
    cout << endl; // Expected: (1, 3) (2, 3)

    nums1 = {1, 1, 2};
    nums2 = {1, 2, 3};
    k = 10;
    result = solution.kSmallestPairs(nums1, nums2, k);
    cout << "Test case 4: ";
    for (auto& pair : result) {
        cout << "(" << pair[0] << ", " << pair[1] << ") ";
    }
    cout << endl; // Expected: (1,1) (1,1) (1,2) (1,2) (2,1) (1,3) (1,3) (2,2) (2,3)

    return 0;
}
// g++ -std=c++17 Leetcode_0373_Find_K_Pairs_with_Smallest_Sums.cpp -o test
