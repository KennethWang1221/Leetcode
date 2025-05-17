#include <iostream>
#include <vector>
#include <numeric>
#include <unordered_map>
#include <algorithm>

using namespace std;

class Graph {
private:
    int n;
    vector<int> parent, compSize;

    int getParent(int x) {
        if (parent[x] == x) {
            return x;
        }
        return parent[x] = getParent(parent[x]);
    }

    void unionSet(int x, int y) {
        int parx = getParent(x), pary = getParent(y);
        if (parx != pary) {
            if (compSize[parx] < compSize[pary]) {
                swap(parx, pary);
            }
            parent[pary] = parx;
            compSize[parx] += compSize[pary];
        }
    }

public:
    Graph(int n = 0) : n(n) {
        parent.resize(n);
        compSize.resize(n, 1);
        iota(parent.begin(), parent.end(), 0);
    }

    void addEdge(int x, int y) {
        unionSet(x, y);
    }

    bool isConnected() {
        return compSize[getParent(0)] == n;
    }
};

class Solution {
private:
    vector<int> getPrimeFactors(int x) {
        vector<int> primeFactors;
        for (int i = 2; i * i <= x; ++i) {
            if (x % i == 0) {
                primeFactors.push_back(i);
                while (x % i == 0) x /= i;
            }
        }
        if (x > 1) primeFactors.push_back(x);
        return primeFactors;
    }

public:
    bool canTraverseAllPaths(vector<int>& nums) {
        int n = nums.size();
        if (n == 1) return true;

        // Edge case: any 1 in array breaks everything
        for (int num : nums) {
            if (num == 1) {
                return false;
            }
        }

        Graph g(n);
        unordered_map<int, int> seen;

        for (int i = 0; i < n; ++i) {
            vector<int> factors = getPrimeFactors(nums[i]);
            for (int f : factors) {
                if (seen.count(f)) {
                    g.addEdge(i, seen[f]);
                } else {
                    seen[f] = i;
                }
            }
        }

        return g.isConnected();
    }
};

// Test Cases
int main() {
    Solution sol;

    vector<int> nums1 = {2,3,6};
    cout << "Test Case 1 ([2,3,6]): " << (sol.canTraverseAllPaths(nums1) ? "true" : "false") << endl; // Expected: true

    vector<int> nums2 = {5,2,3,8,10};
    cout << "Test Case 2 ([5,2,3,8,10]): " << (sol.canTraverseAllPaths(nums2) ? "true" : "false") << endl; // Expected: false

    vector<int> nums3 = {6,6,6,6};
    cout << "Test Case 3 ([6,6,6,6]): " << (sol.canTraverseAllPaths(nums3) ? "true" : "false") << endl; // Expected: true

    vector<int> nums4 = {1,1};
    cout << "Test Case 4 ([1,1]): " << (sol.canTraverseAllPaths(nums4) ? "true" : "false") << endl; // Expected: false

    vector<int> nums5 = {1};
    cout << "Test Case 5 ([1]): " << (sol.canTraverseAllPaths(nums5) ? "true" : "false") << endl; // Expected: true

    vector<int> nums6 = {7, 49, 343};
    cout << "Test Case 6 ([7,49,343]): " << (sol.canTraverseAllPaths(nums6) ? "true" : "false") << endl; // Expected: true

    vector<int> nums7 = {2,4,8,16,32};
    cout << "Test Case 7 ([2,4,8,16,32]): " << (sol.canTraverseAllPaths(nums7) ? "true" : "false") << endl; // Expected: true

    vector<int> nums8 = {2,3,5,7,11};
    cout << "Test Case 8 ([2,3,5,7,11]): " << (sol.canTraverseAllPaths(nums8) ? "true" : "false") << endl; // Expected: true

    vector<int> nums9 = {2,3,5,7,13,17,29,31};
    cout << "Test Case 9 ([2,3,5,7,13,17,29,31]): " << (sol.canTraverseAllPaths(nums9) ? "true" : "false") << endl; // Expected: true

    vector<int> nums10 = {2,3,5,7,11,13,17,23,29,31,37};
    cout << "Test Case 10 ([2,3,5,7,11,13,17,23,29,31,37]): " << (sol.canTraverseAllPaths(nums10) ? "true" : "false") << endl; // Expected: true

    return 0;
}