#include <iostream>
#include <vector>
#include <algorithm>

class MyHashSet {
private:
    std::vector<int> arr;

public:
    MyHashSet() {
        // Initialize the set (empty vector)
    }

    void add(int key) {
        // If the key is not already in the set, add it
        if (std::find(arr.begin(), arr.end(), key) == arr.end()) {
            arr.push_back(key);
        }
    }

    void remove(int key) {
        // If the key exists in the set, remove it
        auto it = std::find(arr.begin(), arr.end(), key);
        if (it != arr.end()) {
            arr.erase(it);
        }
    }

    bool contains(int key) {
        // Check if the key exists in the set
        return std::find(arr.begin(), arr.end(), key) != arr.end();
    }
};

int main() {
    MyHashSet obj;
    obj.add(1);
    obj.add(2);
    std::cout << "Contains 1: " << obj.contains(1) << std::endl; // Expected output: 1 (true)
    std::cout << "Contains 3: " << obj.contains(3) << std::endl; // Expected output: 0 (false)
    obj.add(2);
    std::cout << "Contains 2: " << obj.contains(2) << std::endl; // Expected output: 1 (true)
    obj.remove(2);
    std::cout << "Contains 2: " << obj.contains(2) << std::endl; // Expected output: 0 (false)

    return 0;
}
// g++ -std=c++17 Leetcode_0705_Design_HashSet.cpp -o test