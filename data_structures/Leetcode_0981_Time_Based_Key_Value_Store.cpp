#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
using namespace std;

class TimeMap {
public:
    // Store the key-value pairs, with the key associated with a list of [value, timestamp]
    unordered_map<string, vector<pair<string, int>>> keyStore;

    TimeMap() {}

    // Set the key-value pair along with the timestamp
    void set(string key, string value, int timestamp) {
        keyStore[key].push_back({value, timestamp});
    }

    // Get the value associated with the key at the specific timestamp
    string get(string key, int timestamp) {
        string res = "";
        auto& values = keyStore[key];
        int l = 0, r = values.size() - 1;

        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (values[mid].second <= timestamp) {
                res = values[mid].first;  // Update the result with the latest value that satisfies the timestamp condition
                l = mid + 1;  // Move to the right part to find the latest value
            } else {
                r = mid - 1;  // Move to the left part to find values before the timestamp
            }
        }

        return res;
    }
};

int main() {
    TimeMap timeMap;

    timeMap.set("foo", "bar", 1);
    cout << timeMap.get("foo", 1) << endl;  // Expected output: "bar"
    cout << timeMap.get("foo", 3) << endl;  // Expected output: "bar"
    timeMap.set("foo", "bar2", 4);
    cout << timeMap.get("foo", 4) << endl;  // Expected output: "bar2"
    cout << timeMap.get("foo", 5) << endl;  // Expected output: "bar2"

    return 0;
}

// g++ -std=c++17 Leetcode_0981_Time_Based_Key_Value_Store.cpp -o test
