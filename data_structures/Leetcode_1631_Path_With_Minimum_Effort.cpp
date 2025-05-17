#include <iostream>
#include <vector>
#include <queue>
#include <cmath>

using namespace std;

class Solution {
public:
    int minimumEffortPath(vector<vector<int>>& heights) {
        int rows = heights.size();
        int cols = heights[0].size();
        
        // Direction vectors for 4 directions (up, down, left, right)
        vector<int> dirs = {-1, 0, 1, 0, -1, 0}; 
        
        auto valid = [&](int mid) {
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
};

int main() {
    Solution sol;
    vector<vector<int>> heights = {
        {1, 2, 3},
        {3, 8, 4},
        {5, 3, 5}
    };
    
    cout << sol.minimumEffortPath(heights) << endl;  // Output the minimum effort path
    
    return 0;
}
// g++ -std=c++17 Leetcode_1631_Path_With_Minimum_Effort.cpp -o test 