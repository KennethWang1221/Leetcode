#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <unordered_set>

using namespace std;

class Solution {
public:
    int minCostConnectPoints(vector<vector<int>>& points) {
        int N = points.size();
        
        // Min-Heap (priority queue) for Prim's algorithm
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minH;
        minH.push({0, 0});  // Start with point 0, cost 0
        
        // Vector to track if a point has been visited
        unordered_set<int> visit;
        int res = 0;
        
        while (visit.size() < N) {
            auto [cost, i] = minH.top();
            minH.pop();
            
            // Skip if the point is already visited
            if (visit.find(i) != visit.end()) {
                continue;
            }

            // Add the cost to the result and mark the point as visited
            res += cost;
            visit.insert(i);

            // Add all the neighbors to the priority queue
            for (int j = 0; j < N; ++j) {
                if (i != j && visit.find(j) == visit.end()) {
                    int dist = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1]);
                    minH.push({dist, j});
                }
            }
        }

        return res;
    }
};

int main() {
    Solution sol;
    vector<vector<int>> points = {{0,0},{2,2},{3,10},{5,2},{7,0}};
    
    int res = sol.minCostConnectPoints(points);
    cout << "Minimum cost to connect all points: " << res << endl;  // Output the result
    
    return 0;
}


// g++ -std=c++17 Leetcode_1584_Min_Cost_to_Connect_All_Points.cpp -o test 