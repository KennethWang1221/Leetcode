#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
        if (n == 1) return {0};  // Special case for a single node

        // Step 1: Build the adjacency list for the graph
        unordered_map<int, unordered_set<int>> adj;
        for (const auto& edge : edges) {
            adj[edge[0]].insert(edge[1]);
            adj[edge[1]].insert(edge[0]);
        }

        // Step 2: Initialize the queue with leaves (nodes with degree 1)
        queue<int> q;
        for (const auto& node : adj) {
            if (node.second.size() == 1) {
                q.push(node.first);
            }
        }

        // Step 3: Perform BFS to remove leaves layer by layer
        while (n > 2) {  // Continue until we are left with 1 or 2 nodes
            int size = q.size();
            n -= size;  // Decrease the number of nodes by the number of leaves
            for (int i = 0; i < size; ++i) {
                int leaf = q.front();
                q.pop();
                for (int neighbor : adj[leaf]) {
                    adj[neighbor].erase(leaf);  // Remove the leaf from its neighbor's adjacency list
                    if (adj[neighbor].size() == 1) {  // If the neighbor has become a leaf, add to the queue
                        q.push(neighbor);
                    }
                }
            }
        }

        // Step 4: The remaining nodes in the queue are the roots of the minimum height trees
        vector<int> result;
        while (!q.empty()) {
            result.push_back(q.front());
            q.pop();
        }

        return result;
    }
};

int main() {
    Solution sol;
    int n = 6;
    vector<vector<int>> edges = {{3, 0}, {3, 1}, {3, 2}, {3, 4}, {5, 4}};

    vector<int> res = sol.findMinHeightTrees(n, edges);
    
    cout << "The root(s) of the minimum height tree(s): ";
    for (int node : res) {
        cout << node << " ";
    }
    cout << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0310_Minimum_Height_Trees.cpp -o test 