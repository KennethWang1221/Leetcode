#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int findMinArrowShots(vector<vector<int>>& points) {
        if (points.empty()) return 0;

        // Sort by ending value (greedy approach)
        sort(points.begin(), points.end(), [](const vector<int>& a, const vector<int>& b) {
            return a[1] < b[1];
        });

        int arrows = 1;
        int end = points[0][1];

        for (int i = 1; i < points.size(); ++i) {
            int start = points[i][0];
            int currEnd = points[i][1];

            if (start > end) {
                arrows++;
                end = currEnd;
            }
        }

        return arrows;
    }
};

// Test Case
int main() {
    Solution sol;

    vector<vector<int>> points1 = {
        {3,9},{7,12},{3,8},{6,8},{9,10},{2,9},{0,9},{3,9},{0,6},{2,8}
    };
    cout << sol.findMinArrowShots(points1) << endl; // Output: 2

    vector<vector<int>> points2 = {{10,16},{2,8},{1,6},{7,12}};
    cout << sol.findMinArrowShots(points2) << endl; // Output: 2

    return 0;
}

// g++ -std=c++17 Leetcode_0452_Minimum_Number_of_Arrows_to_Burst_Balloons.cpp -o test