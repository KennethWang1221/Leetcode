#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>

using namespace std;

class Solution {
public:
    // BFS function to find the result of a query
    double bfs(const string& src, const string& target, unordered_map<string, vector<pair<string, double>>>& adj) {
        if (adj.find(src) == adj.end() || adj.find(target) == adj.end()) {
            return -1.0;  // If src or target is not in the graph, return -1
        }

        queue<pair<string, double>> q;  // Queue for BFS
        unordered_set<string> visited;  // Set to track visited nodes
        q.push({src, 1.0});  // Start with the source node and weight 1
        visited.insert(src);

        while (!q.empty()) {
            auto [node, weight] = q.front();
            q.pop();

            // If the target node is found, return the weight
            if (node == target) {
                return weight;
            }

            // Explore neighbors
            for (auto& neighbor : adj[node]) {
                string nei = neighbor.first;
                double newWeight = weight * neighbor.second;

                if (visited.find(nei) == visited.end()) {
                    visited.insert(nei);
                    q.push({nei, newWeight});
                }
            }
        }

        return -1.0;  // Return -1 if no path is found
    }

    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
        unordered_map<string, vector<pair<string, double>>> adj;  // Graph representation

        // Build the graph based on equations and values
        for (int i = 0; i < equations.size(); i++) {
            string a = equations[i][0], b = equations[i][1];
            double val = values[i];

            adj[a].push_back({b, val});
            adj[b].push_back({a, 1.0 / val});
        }

        vector<double> res;
        for (auto& query : queries) {
            string src = query[0], target = query[1];
            res.push_back(bfs(src, target, adj));  // Process each query with BFS
        }

        return res;
    }
};

int main() {
    Solution sol;
    vector<vector<string>> equations = {{"a", "b"}, {"b", "c"}};
    vector<double> values = {2.0, 3.0};
    vector<vector<string>> queries = {{"a", "c"}, {"b", "a"}, {"a", "e"}, {"a", "a"}, {"x", "x"}};

    vector<double> res = sol.calcEquation(equations, values, queries);
    
    // Output the results of the queries
    for (double result : res) {
        cout << result << " ";
    }
    cout << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0399_Evaluate_Division.cpp -o test