#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
using namespace std;

class Solution {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int k) {
        vector<vector<int>> res;
        priority_queue<pair<double, vector<int>>> maxheap;  // Max-heap to store the points and their distances

        for (const auto& p : points) {
            // Calculate the distance from the origin (0, 0)
            double dist = sqrt(p[0] * p[0] + p[1] * p[1]);

            // Push the distance and point into the max-heap
            maxheap.push({dist, p});

            // If the heap size exceeds k, pop the farthest point
            if (maxheap.size() > k) {
                maxheap.pop();
            }
        }

        // Extract the k closest points from the max-heap
        while (!maxheap.empty()) {
            res.push_back(maxheap.top().second);
            maxheap.pop();
        }

        return res;
    }
};

int main() {
    Solution solution;
    vector<vector<int>> points = {{1, 3}, {-2, 2}};
    int k = 1;
    vector<vector<int>> res = solution.kClosest(points, k);

    // Print the result
    for (const auto& point : res) {
        cout << "[" << point[0] << ", " << point[1] << "] ";
    }
    cout << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0973_K_Closest_Points_to_Origin.cpp -o test