#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    // Helper function to find the representative (root) of a node with path compression
    int find(int node, vector<int>& par) {
        if (par[node] != node) {
            par[node] = find(par[node], par);  // Path compression
        }
        return par[node];
    }

    // Helper function to union two sets, returns false if they are already connected
    bool unionSets(int n1, int n2, vector<int>& par, vector<int>& rank) {
        int p1 = find(n1, par);
        int p2 = find(n2, par);

        if (p1 == p2) {
            return false;  // Already in the same set, meaning it's a redundant connection
        }

        // Union by rank: attach the smaller tree to the larger tree
        if (rank[p1] > rank[p2]) {
            par[p2] = p1;
            rank[p1] += rank[p2];
        } else {
            par[p1] = p2;
            rank[p2] += rank[p1];
        }

        return true;
    }

    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        vector<int> par(n + 1), rank(n + 1, 1);

        // Initialize parent array where each node is its own parent initially
        for (int i = 1; i <= n; i++) {
            par[i] = i;
        }

        // Process each edge and try to union
        for (const auto& edge : edges) {
            int n1 = edge[0], n2 = edge[1];
            if (!unionSets(n1, n2, par, rank)) {
                return edge;  // Found the redundant connection
            }
        }

        return {};  // No redundant connection found
    }
};

int main() {
    Solution sol;
    vector<vector<int>> edges = {{1, 2}, {1, 3}, {2, 3}};

    vector<int> res = sol.findRedundantConnection(edges);
    cout << "Redundant connection: [" << res[0] << ", " << res[1] << "]" << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0684_Redundant_Connection.cpp -o test