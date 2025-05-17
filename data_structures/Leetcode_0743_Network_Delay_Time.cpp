#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <climits>

using namespace std;

class Solution {
public:
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        // Step 1: Build the adjacency list
        unordered_map<int, vector<pair<int, int>>> edges;  // node -> [(neighbor, weight)]
        for (const auto& time : times) {
            edges[time[0]].emplace_back(time[1], time[2]);
        }

        // Step 2: Min-Heap (priority queue) for Dijkstra's algorithm
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minHeap;
        minHeap.push({0, k});  // (distance, node)
        
        vector<int> dist(n + 1, INT_MAX);  // Store the shortest time to reach each node
        dist[k] = 0;  // The starting node has a time of 0

        // Step 3: Dijkstra's algorithm to calculate shortest times
        while (!minHeap.empty()) {
            auto [w1, n1] = minHeap.top();
            minHeap.pop();

            // If we have already visited this node, skip it
            if (w1 > dist[n1]) continue;

            // Explore the neighbors of the current node
            for (const auto& [n2, w2] : edges[n1]) {
                int newDist = w1 + w2;
                if (newDist < dist[n2]) {
                    dist[n2] = newDist;
                    minHeap.push({newDist, n2});
                }
            }
        }

        // Step 4: Find the maximum time taken by all nodes to receive the signal
        int res = 0;
        for (int i = 1; i <= n; i++) {
            if (dist[i] == INT_MAX) {
                return -1;  // If any node is unreachable, return -1
            }
            res = max(res, dist[i]);  // Find the maximum time
        }

        return res;
    }
};

int main() {
    Solution sol;
    vector<vector<int>> times = {{1, 3, 1}, {1, 2, 4}, {3, 4, 1}, {4, 2, 1}};
    int n = 4;
    int k = 1;

    int result = sol.networkDelayTime(times, n, k);
    cout << "Network delay time: " << result << endl;  // Output: 4

    return 0;
}


// g++ -std=c++17 Leetcode_0743_Network_Delay_Time.cpp -o test 