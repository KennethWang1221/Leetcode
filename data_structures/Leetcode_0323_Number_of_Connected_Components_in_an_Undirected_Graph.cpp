#include <iostream>
#include <vector>
#include <unordered_set>

using namespace std;

class Solution {
public:
    // Helper DFS function to traverse and mark all connected nodes
    void dfs(int node, vector<vector<int>>& graph, unordered_set<int>& visited) {
        visited.insert(node);  // Mark the current node as visited
        
        // Visit all unvisited neighbors
        for (int neighbor : graph[node]) {
            if (visited.find(neighbor) == visited.end()) {
                dfs(neighbor, graph, visited);
            }
        }
    }

    int countComponents(int n, vector<vector<int>>& edges) {
        // Step 1: Build the adjacency list (graph representation)
        vector<vector<int>> graph(n);
        for (const auto& edge : edges) {
            graph[edge[0]].push_back(edge[1]);
            graph[edge[1]].push_back(edge[0]);  // Because the graph is undirected
        }

        // Step 2: Use DFS to find connected components
        unordered_set<int> visited;
        int components = 0;

        // Step 3: Traverse each node
        for (int i = 0; i < n; i++) {
            if (visited.find(i) == visited.end()) {  // If the node is unvisited
                dfs(i, graph, visited);  // Perform DFS to mark all connected nodes
                components++;  // One more component found
            }
        }

        return components;
    }
};

int main() {
    Solution sol;
    int n = 5;
    vector<vector<int>> edges = {{0, 1}, {1, 2}, {3, 4}};
    
    int res = sol.countComponents(n, edges);
    cout << "Number of connected components: " << res << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0323_Number_of_Connected_Components_in_an_Undirected_Graph.cpp -o test