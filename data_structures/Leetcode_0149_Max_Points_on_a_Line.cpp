#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
using namespace std;

class Solution {
public:
    int maxPoints(vector<vector<int>>& points) {
        int n = points.size();
        if (n <= 1) 
            return n;
        int res = 1;

        for (int i = 0; i < n; i++) {
            int duplicates = 0;
            map<pair<int, int>, int> slope_count;
            int max_count = 0;

            for (int j = i + 1; j < n; j++) {
                if (points[i][0] == points[j][0] && points[i][1] == points[j][1]) {
                    duplicates++;
                } else {
                    int dx = points[j][0] - points[i][0];
                    int dy = points[j][1] - points[i][1];
                    pair<int, int> key;
                    if (dx == 0) {
                        key = make_pair(0, 1);
                    } else if (dy == 0) {
                        key = make_pair(1, 0);
                    } else {
                        int g = gcd(abs(dx), abs(dy));
                        dx /= g;
                        dy /= g;
                        if (dx < 0) {
                            dx = -dx;
                            dy = -dy;
                        }
                        key = make_pair(dx, dy);
                    }
                    slope_count[key]++;
                }
            }

            for (auto it = slope_count.begin(); it != slope_count.end(); it++) {
                if (it->second > max_count) {
                    max_count = it->second;
                }
            }

            res = max(res, 1 + duplicates + max_count);
        }

        return res;
    }

private:
    int gcd(int a, int b) {
        while (b) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }
};

int main() {
    Solution sol;
    vector<vector<int>> points = {{1,1}, {3,2}, {5,3}, {4,1}, {2,3}, {1,4}};
    int result = sol.maxPoints(points);
    cout << "Test case 1: " << result << endl;  // Expected: 4

    points = {{0,0}, {1,1}, {0,0}};
    result = sol.maxPoints(points);
    cout << "Test case 2: " << result << endl;  // Expected: 3

    points = {{1,1}};
    result = sol.maxPoints(points);
    cout << "Test case 3: " << result << endl;  // Expected: 1

    return 0;
}

// g++ -std=c++17 Leetcode_0149_Max_Points_on_a_Line.cpp -o test