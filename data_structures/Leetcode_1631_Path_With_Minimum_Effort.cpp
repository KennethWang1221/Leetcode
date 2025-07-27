#include <iostream>
#include <vector>
#include <queue>
#include <cmath>

using namespace std;

class Solution {
public:
    int minimumEffortPath_BFS(vector<vector<int>>& heights) {
        int rows = heights.size();
        int cols = heights[0].size();
        
        // Direction vectors for 4 directions (up, down, left, right)
        vector<int> dirs = {-1, 0, 1, 0, -1, 0}; 
        
        auto valid = [&](int mid) -> bool {
            // BFS to check if we can reach the bottom-right corner with maximum effort <= mid
            vector<vector<bool>> visited(rows, vector<bool>(cols, false));
            queue<pair<int, int>> q;
            q.push({0, 0});
            visited[0][0] = true;

            while (!q.empty()) {
                auto [r, c] = q.front();
                q.pop();

                // If we've reached the bottom-right corner
                if (r == rows - 1 && c == cols - 1) return true;

                // Explore all four directions
                for (int i = 0; i < 4; i++) {
                    int nr = r + dirs[i], nc = c + dirs[i + 1];
                    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !visited[nr][nc]) {
                        // If the effort to move to this cell is less than or equal to mid, we visit it
                        if (abs(heights[r][c] - heights[nr][nc]) <= mid) {
                            visited[nr][nc] = true;
                            q.push({nr, nc});
                        }
                    }
                }
            }

            return false;
        };
        
        // Binary search on the maximum effort
        int left = 0, right = 1000000, answer = 0;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (valid(mid)) {
                answer = mid;  // If we can reach the destination with mid effort, try lower values
                right = mid - 1;
            } else {
                left = mid + 1;  // Otherwise, try higher values
            }
        }
        
        return answer;
    }

    int minimumEffortPath_min_heap(vector<vector<int>>& heights) {
        int rows = heights.size();
        int cols = heights[0].size();

        // Directions for moving up, down, left, and right
        vector<vector<int>> dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        // Min-Heap for Dijkstra's algorithm, stores {effort, x, y}
        priority_queue<vector<int>, vector<vector<int>>, greater<vector<int>>> min_heap;
        min_heap.push({0, 0, 0});  // Starting at (0, 0) with effort 0
        
        vector<vector<bool>> visited(rows, vector<bool>(cols, false));  // To keep track of visited cells

        while (!min_heap.empty()) {
            vector<int> current = min_heap.top();
            min_heap.pop();
            
            int effort = current[0];
            int x = current[1];
            int y = current[2];

            // Skip if already visited
            if (visited[x][y]) {
                continue;
            }

            // Mark as visited when processing
            visited[x][y] = true;

            // If we reach the bottom-right corner, return the effort
            if (x == rows - 1 && y == cols - 1) {
                return effort;
            }

            // Explore the 4 neighboring cells
            for (const auto& dir : dirs) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                
                // Check bounds and if not visited
                if (nx < 0 || nx >= rows || ny < 0 || ny >= cols || visited[nx][ny]) {
                    continue;
                }

                // Calculate the effort to reach the neighboring cell
                int new_effort = max(effort, abs(heights[x][y] - heights[nx][ny]));
                min_heap.push({new_effort, nx, ny});
            }
        }

        return 0;  // This should never be reached if a path exists
    }
};

int main() {
    Solution sol;
    vector<vector<int>> heights = {
        {1, 2, 3},
        {3, 8, 4},
        {5, 3, 5}
    };
    
    cout << sol.minimumEffortPath_BFS(heights) << endl;  // Output the minimum effort path
    cout << sol.minimumEffortPath_min_heap(heights) << endl;
    return 0;
}
// g++ -std=c++17 Leetcode_1631_Path_With_Minimum_Effort.cpp -o test 
