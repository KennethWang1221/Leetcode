#include <iostream>
#include <vector>
using namespace std;

// Definition for a QuadTree node
class Node {
public:
    bool val;
    bool isLeaf;
    Node* topLeft;
    Node* topRight;
    Node* bottomLeft;
    Node* bottomRight;

    Node() : val(false), isLeaf(false), topLeft(nullptr), topRight(nullptr), bottomLeft(nullptr), bottomRight(nullptr) {}
    Node(bool _val, bool _isLeaf) : val(_val), isLeaf(_isLeaf), topLeft(nullptr), topRight(nullptr), bottomLeft(nullptr), bottomRight(nullptr) {}
    Node(bool _val, bool _isLeaf, Node* _topLeft, Node* _topRight, Node* _bottomLeft, Node* _bottomRight) 
        : val(_val), isLeaf(_isLeaf), topLeft(_topLeft), topRight(_topRight), bottomLeft(_bottomLeft), bottomRight(_bottomRight) {}
};

class Solution {
public:
    Node* construct(vector<vector<int>>& grid) {
        return dfs(grid, 0, 0, grid.size());
    }

    Node* dfs(vector<vector<int>>& grid, int r, int c, int n) {
        bool allSame = true;
        
        // Check if all values in the grid section are the same
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[r + i][c + j] != grid[r][c]) {
                    allSame = false;
                    break;
                }
            }
            if (!allSame) break;
        }
        
        // If all values are the same, return a leaf node
        if (allSame) {
            return new Node(grid[r][c] == 1, true);
        }
        
        // Otherwise, split the grid into four parts and recurse
        n /= 2;
        Node* topLeft = dfs(grid, r, c, n);
        Node* topRight = dfs(grid, r, c + n, n);
        Node* bottomLeft = dfs(grid, r + n, c, n);
        Node* bottomRight = dfs(grid, r + n, c + n, n);
        
        return new Node('*', false, topLeft, topRight, bottomLeft, bottomRight);
    }
};

// Helper function to print the QuadTree
void printQuadTree(Node* node) {
    if (node->isLeaf) {
        cout << "Leaf: " << (node->val ? 1 : 0) << endl;
    } else {
        cout << "Internal:" << endl;
        if (node->topLeft) printQuadTree(node->topLeft);
        if (node->topRight) printQuadTree(node->topRight);
        if (node->bottomLeft) printQuadTree(node->bottomLeft);
        if (node->bottomRight) printQuadTree(node->bottomRight);
    }
}

int main() {
    Solution sol;
    vector<vector<int>> grid = {{1, 1, 0, 0}, {1, 1, 0, 0}, {1, 1, 1, 1}, {1, 1, 1, 1}};
    
    // Construct the QuadTree
    Node* quadTree = sol.construct(grid);
    
    // Print the QuadTree
    printQuadTree(quadTree);
    
    return 0;
}


// g++ -std=c++17 Leetcode_0427_Construct_Quad_Tree.cpp -o test
