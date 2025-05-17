#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class UnionFind {
public:
    UnionFind(int n) {
        par.resize(n);
        rank.resize(n, 1);
        for (int i = 0; i < n; ++i) {
            par[i] = i;
        }
    }

    int find(int v1) {
        if (v1 != par[v1]) {
            par[v1] = find(par[v1]);
        }
        return par[v1];
    }

    bool unionSets(int v1, int v2) {
        int p1 = find(v1), p2 = find(v2);
        if (p1 == p2) {
            return false;
        }

        if (rank[p1] > rank[p2]) {
            par[p2] = p1;
            rank[p1] += rank[p2];
        } else {
            par[p1] = p2;
            rank[p2] += rank[p1];
        }
        return true;
    }

private:
    vector<int> par, rank;
};

class Solution {
public:
    vector<vector<int>> findCriticalAndPseudoCriticalEdges(int n, vector<vector<int>>& edges) {
        // Add the index to each edge
        for (int i = 0; i < edges.size(); ++i) {
            edges[i].push_back(i); // [v1, v2, weight, original_index]
        }

        // Sort the edges based on their weight
        sort(edges.begin(), edges.end(), [](const vector<int>& a, const vector<int>& b) {
            return a[2] < b[2];  // Compare by weight
        });

        // Calculate the MST weight using Kruskal's algorithm
        int mstWeight = 0;
        UnionFind uf(n);
        for (const auto& edge : edges) {
            int v1 = edge[0], v2 = edge[1], weight = edge[2];
            if (uf.unionSets(v1, v2)) {
                mstWeight += weight;
            }
        }

        vector<int> critical, pseudo;

        // Check each edge for being critical or pseudo-critical
        for (int i = 0; i < edges.size(); ++i) {
            int v1 = edges[i][0], v2 = edges[i][1], weight = edges[i][2], index = edges[i][3];

            // Try to build MST without this edge (check if it is critical)
            UnionFind ufWithoutEdge(n);
            int weightWithoutEdge = 0;
            for (int j = 0; j < edges.size(); ++j) {
                if (i != j) {
                    int u1 = edges[j][0], u2 = edges[j][1], w = edges[j][2];
                    if (ufWithoutEdge.unionSets(u1, u2)) {
                        weightWithoutEdge += w;
                    }
                }
            }

            if (ufWithoutEdge.find(v1) != ufWithoutEdge.find(v2) || weightWithoutEdge > mstWeight) {
                critical.push_back(index);
                continue;
            }

            // Try to build MST with this edge (check if it is pseudo-critical)
            UnionFind ufWithEdge(n);
            ufWithEdge.unionSets(v1, v2);
            int weightWithEdge = weight;
            for (int j = 0; j < edges.size(); ++j) {
                if (i != j) {
                    int u1 = edges[j][0], u2 = edges[j][1], w = edges[j][2];
                    if (ufWithEdge.unionSets(u1, u2)) {
                        weightWithEdge += w;
                    }
                }
            }

            if (weightWithEdge == mstWeight) {
                pseudo.push_back(index);
            }
        }

        return {critical, pseudo};
    }
};

int main() {
    Solution sol;
    int n = 5;
    vector<vector<int>> edges = {
        {0, 1, 1}, {1, 2, 1}, {2, 3, 2}, {0, 3, 2}, {0, 4, 3}, {3, 4, 3}, {1, 4, 6}
    };
    
    vector<vector<int>> result = sol.findCriticalAndPseudoCriticalEdges(n, edges);
    
    cout << "Critical edges: ";
    for (int index : result[0]) {
        cout << index << " ";
    }
    cout << endl;
    
    cout << "Pseudo-critical edges: ";
    for (int index : result[1]) {
        cout << index << " ";
    }
    cout << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_1489_Find_Critical_and_Pseudo_Critical_Edges_in_Minimum_Spanning_Tree.cpp -o test 