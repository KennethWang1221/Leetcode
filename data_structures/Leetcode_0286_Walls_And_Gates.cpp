#include <iostream>
#include <vector>
#include <queue>
#include <set>

using namespace std;

void addRoom(int r, int c, int rows, int cols, set<pair<int, int>>& visit, queue<pair<int, int>>& q, vector<vector<int>>& rooms) {
    if (r < 0 || c < 0 || r >= rows || c >= cols || visit.count({r, c}) || rooms[r][c] == -1) {
        return;
    }
    visit.insert({r, c});
    q.push({r, c});
}

void wallsAndGates(vector<vector<int>>& rooms) {
    int rows = rooms.size();
    int cols = rooms[0].size();
    set<pair<int, int>> visit;
    queue<pair<int, int>> q;

    // Add all gates (rooms with value 0) to the queue
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (rooms[r][c] == 0) {
                q.push({r, c});
                visit.insert({r, c});
            }
        }
    }

    int dist = 0;
    // Perform BFS
    while (!q.empty()) {
        int n = q.size();
        for (int i = 0; i < n; i++) {
            auto [r, c] = q.front();
            q.pop();
            rooms[r][c] = dist;

            // Explore all four directions
            addRoom(r + 1, c, rows, cols, visit, q, rooms);
            addRoom(r - 1, c, rows, cols, visit, q, rooms);
            addRoom(r, c + 1, rows, cols, visit, q, rooms);
            addRoom(r, c - 1, rows, cols, visit, q, rooms);
        }
        dist++;
    }
}

int main() {
    vector<vector<int>> rooms = {
        {2147483647, -1, 0, 2147483647},
        {2147483647, 2147483647, 2147483647, -1},
        {2147483647, -1, 2147483647, -1},
        {0, -1, 2147483647, 2147483647}
    };
    
    wallsAndGates(rooms);

    // Print the resulting rooms matrix
    for (const auto& row : rooms) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    return 0;
}


// g++ -std=c++17 Leetcode_0286_Walls_And_Gates.cpp -o test