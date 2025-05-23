// Full testable code
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>

using namespace std;

struct PairHash {
    template <class T1, class T2>
    size_t operator()(const pair<T1, T2>& p) const {
        return hash<T1>()(p.first) ^ hash<T2>()(p.second);
    }
};

class DetectSquares {
private:
    unordered_map<pair<int, int>, int, PairHash> ptsCount;
    vector<pair<int, int>> pts;

public:
    DetectSquares() {}

    void add(vector<int> point) {
        pair<int, int> p = {point[0], point[1]};
        ptsCount[p]++;
        pts.push_back(p);
    }

    int count(vector<int> point) {
        int px = point[0];
        int py = point[1];
        int res = 0;

        for (auto& p : pts) {
            int x = p.first;
            int y = p.second;

            if (abs(py - y) != abs(px - x) || x == px || y == py)
                continue;

            res += ptsCount[{x, py}] * ptsCount[{px, y}];
        }

        return res;
    }
};

// Test Case
int main() {
    DetectSquares ds;

    ds.add({3, 10});
    ds.add({11, 2});
    ds.add({3, 2});
    ds.add({11, 10});
    ds.add({11, 10});

    cout << ds.count({11, 2}) << endl; // Output: 1
    cout << ds.count({14, 10}) << endl; // Output: 0

    return 0;
}

// g++ -std=c++17 Leetcode_2013_Detect_Squares.cpp -o test
